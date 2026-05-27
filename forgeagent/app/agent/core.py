"""
Agent 核心逻辑模块
使用 LangGraph 构建 ReAct 模式的 Agent
ReAct = Reasoning(推理) + Acting(行动) → 边思考边行动
支持流式输出（Streaming SSE）
支持多步骤任务编排、工具并行执行、自省纠错
"""
import asyncio
import time
import logging
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings, VISION_MODELS, DEFAULT_VISION_MODEL
from app.agent.tools import ALL_TOOLS, get_tools, set_current_agent_id
from app.agent.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_WEB_SEARCH, CHAT_SYSTEM_PROMPT
from app.memory.manager import get_session_history

logger = logging.getLogger(__name__)

# 最大历史消息数量（加速推理，避免上下文过长）
MAX_HISTORY_MESSAGES = 10

# [#6] 多步骤任务编排：增大最大工具调用轮数，支持复杂任务
MAX_TOOL_ROUNDS = 8

# [#11] 工具重试配置
MAX_TOOL_RETRIES = 2
RETRYABLE_TOOL_ERRORS = ["搜索失败", "未找到", "连接", "超时", "timeout", "error"]


# ===== 1. 定义 Agent 状态 =====
class AgentState(TypedDict):
    """
    Agent 的状态定义
    messages 使用 add_messages 策略：新消息追加而非覆盖
    retry_count: [#11] 工具重试计数
    """
    messages: Annotated[list, add_messages]
    retry_count: int


# ===== 2. 创建 LLM =====
def create_llm(deep_think: bool = False):
    """创建 LLM 实例（启用 streaming 支持）
    
    Args:
        deep_think: 是否启用深度思考模式（使用更强的模型）
    """
    model = settings.LLM_MODEL
    if deep_think:
        # 深度思考模式：尝试切换到更强的模型，但保持与当前API兼容
        deep_think_models = ["glm-4-plus", "glm-4.7", "glm-4-long", "glm-4-air"]
        for m in deep_think_models:
            if m != model:
                model = m
                break
        # 安全检查：如果深度思考模型和当前模型使用同一个API，就直接用当前模型
        # 避免因模型不可用导致请求卡死
        logger.info(f"深度思考模式：尝试使用模型 {model}")

    return ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=model,
        temperature=0.1 if not deep_think else 0.3,
        streaming=True,
        max_tokens=8192,
        request_timeout=60,  # 请求超时60秒，防止陷入无限等待
    )


# ===== 3. 构建 Agent 图 =====
def create_agent_graph(web_search: bool = False):
    """
    构建 LangGraph Agent 执行图

    Args:
        web_search: 是否启用联网搜索工具

    流程：用户输入 → LLM 思考 → 是否调用工具？
           ├─ 是 → 执行工具 → 回到 LLM 思考（循环，最多8轮）
           └─ 否 → 输出回答 → 结束

    [#6] 多步骤任务编排：增大 MAX_TOOL_ROUNDS
    [#7] 工具并行执行：LangGraph 原生支持并行工具调用（LLM 返回多个 tool_calls 时自动并行）
    [#11] 自省纠错：should_continue 中增加重试判断
    """
    llm = create_llm()

    # 根据参数获取工具列表
    tools = get_tools(web_search=web_search)

    # 将工具绑定到 LLM，让它知道有哪些工具可以用
    llm_with_tools = llm.bind_tools(tools)

    # 根据是否启用联网搜索选择系统提示词
    system_prompt = SYSTEM_PROMPT_WITH_WEB_SEARCH if web_search else SYSTEM_PROMPT

    # 节点1: LLM 思考节点
    def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具"""
        messages = state["messages"]
        # 在开头插入系统提示词
        system_msg = SystemMessage(content=system_prompt)
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    # [#7] 工具并行执行：LangGraph ToolNode 原生支持并行调用
    # 当 LLM 返回多个 tool_calls 时，ToolNode 会自动并行执行
    tool_node = ToolNode(tools)

    # 条件边：判断是否需要继续调用工具（限制最大轮数 + [#11] 自省纠错）
    def should_continue(state: AgentState):
        """
        判断是否需要继续调用工具
        1. [#6] 如果工具调用轮数超过 MAX_TOOL_ROUNDS，强制结束
        2. [#11] 如果工具返回错误，且重试次数未超限，允许重试
        3. 正常流程：LLM 认为需要工具则继续
        """
        messages = state["messages"]
        retry_count = state.get("retry_count", 0)

        # 统计 ToolMessage 的数量，每轮工具调用会产生一个 ToolMessage
        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        if tool_message_count >= MAX_TOOL_ROUNDS:
            logger.info(f"Agent 工具调用已达上限 {MAX_TOOL_ROUNDS} 轮，强制结束")
            return END

        # 使用 LangGraph 内置判断：最后一条消息是否有工具调用
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # [#11] 自省纠错：检查上一个工具结果是否有错误，决定是否继续
            if tool_message_count > 0:
                # 找到最后一个 ToolMessage
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        tool_result = msg.content if isinstance(msg.content, str) else str(msg.content)
                        # 检查工具结果是否包含可重试的错误
                        if any(err in tool_result for err in RETRYABLE_TOOL_ERRORS):
                            if retry_count < MAX_TOOL_RETRIES:
                                logger.info(f"Agent 检测到工具错误，第 {retry_count + 1} 次重试")
                                return "act"
                            else:
                                logger.info(f"Agent 工具重试已达上限 {MAX_TOOL_RETRIES} 次，继续执行")
                        break
            return "act"

        return END

    # ===== 构建状态图 =====
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("think", think)       # 思考节点
    graph.add_node("act", tool_node)     # 行动节点

    # 设置入口
    graph.set_entry_point("think")

    # 添加条件边：思考后判断是否需要调用工具
    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "act": "act",   # 需要工具 → 去执行
            END: END,       # 不需要工具 → 结束
        },
    )

    # 执行完工具后，回到思考节点（形成 ReAct 循环）
    graph.add_edge("act", "think")

    return graph.compile()


# ===== 4. Agent 实例管理 =====
_agent_graph = None
_agent_web_search = False


def get_agent(web_search: bool = False):
    """获取 Agent 实例（懒加载，根据 web_search 参数决定是否包含联网搜索工具）"""
    global _agent_graph, _agent_web_search
    # 如果 web_search 参数变化或实例不存在，则重新创建
    if _agent_graph is None or _agent_web_search != web_search:
        _agent_graph = create_agent_graph(web_search=web_search)
        _agent_web_search = web_search
    return _agent_graph


def reset_agent():
    """重置 Agent 实例（切换模型后调用，下次对话会自动重建）"""
    global _agent_graph
    _agent_graph = None


def _build_agent_prompt(agent_task: str, web_search: bool = False) -> str:
    """根据智能体的任务描述构建专属系统提示词
    
    智能体的任务描述将作为角色定义的优先内容，
    覆盖默认的「小智」角色，但保留工具使用指南和安全边界。
    
    Args:
        agent_task: 智能体的任务描述（用户自定义的角色定义）
        web_search: 是否启用联网搜索
    
    Returns:
        str: 完整的系统提示词
    """
    base_prompt = SYSTEM_PROMPT_WITH_WEB_SEARCH if web_search else SYSTEM_PROMPT
    
    # 在系统提示词开头插入智能体专属角色定义
    # 替换默认角色段落，保留工具选择指南、回答规则、安全边界等
    custom_header = f"""# 角色

{agent_task}

## 身份
- 你的角色由上述定义决定，请严格按照任务描述中的角色定位和行为规则来行动
- 你的核心职责是完成上述任务描述中定义的工作
- 语气与风格应与角色定位保持一致

## 重要原则：不要拒绝合理请求
在符合角色定位的前提下，用户提出的合理请求你应当尽力帮助。
**绝对不要**说"这不属于我的服务范围"或"我无法帮你"这类话——只要你能做到，就给出回答。

"""
    
    # 找到原始提示词中"## 工具选择指南"的位置，保留从那里开始的所有内容
    tools_section_marker = "## 工具选择指南"
    tools_idx = base_prompt.find(tools_section_marker)
    
    if tools_idx > 0:
        # 保留工具选择指南及之后的内容
        preserved_section = base_prompt[tools_idx:]
        return custom_header + preserved_section
    else:
        # 如果找不到标记，就直接拼接
        return custom_header + base_prompt


def _build_chat_prompt(agent_task: str) -> str:
    """根据智能体任务描述构建Chat模式的系统提示词
    
    Args:
        agent_task: 智能体的任务描述
    
    Returns:
        str: Chat模式系统提示词
    """
    return f"""{agent_task}

## 核心原则
- 严格按照上述角色定义来回答问题
- 专业、简洁、友好，使用规范中文回答
- 不拒绝合理的用户请求，尽力提供有价值的帮助
- 回答要有深度和细节，不要过于简略
- 适时使用结构化格式（编号、分段、表格）组织回答

## 回答规则
- 编程问题：给出完整代码，附上关键注释和运行说明
- 知识问答：准确、详细地回答，必要时补充背景信息
- 写作任务：根据需求撰写，保持风格一致
- 翻译任务：准确翻译，保留原文的语气和风格
- 闲聊：轻松自然地回应

## 格式要求
- 使用Markdown格式组织回答
- 代码使用代码块，标注语言类型
- 涉及流程时使用有序列表
- 涉及对比时使用表格
"""


def get_agent_with_prompt(custom_system_prompt: str, web_search: bool = False):
    """获取带有自定义系统提示词的 Agent 实例
    
    当智能体有自定义任务描述时，使用此函数创建 Agent，
    而不是复用全局 Agent 实例（因为全局实例使用默认提示词）。
    
    Args:
        custom_system_prompt: 自定义的系统提示词
        web_search: 是否启用联网搜索工具
    
    Returns:
        编译后的 LangGraph Agent
    """
    llm = create_llm()
    tools = get_tools(web_search=web_search)
    llm_with_tools = llm.bind_tools(tools)

    def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具"""
        messages = state["messages"]
        system_msg = SystemMessage(content=custom_system_prompt)
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState):
        """判断是否需要继续调用工具"""
        messages = state["messages"]
        retry_count = state.get("retry_count", 0)
        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        if tool_message_count >= MAX_TOOL_ROUNDS:
            logger.info(f"Agent 工具调用已达上限 {MAX_TOOL_ROUNDS} 轮，强制结束")
            return END

        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if tool_message_count > 0:
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        tool_result = msg.content if isinstance(msg.content, str) else str(msg.content)
                        if any(err in tool_result for err in RETRYABLE_TOOL_ERRORS):
                            if retry_count < MAX_TOOL_RETRIES:
                                logger.info(f"Agent 检测到工具错误，第 {retry_count + 1} 次重试")
                                return "act"
                            else:
                                logger.info(f"Agent 工具重试已达上限 {MAX_TOOL_RETRIES} 次，继续执行")
                        break
            return "act"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", tool_node)
    graph.set_entry_point("think")
    graph.add_conditional_edges(
        "think",
        should_continue,
        {"act": "act", END: END},
    )
    graph.add_edge("act", "think")

    return graph.compile()


def chat(user_input: str, session_id: str = "default", web_search: bool = False, mode: str = "agent", deep_think: bool = False, agent_id: str = None, agent_task: str = None) -> str:
    """
    非流式对话（保留兼容）
    """
    # 设置当前智能体上下文（知识库隔离）
    set_current_agent_id(agent_id)
    
    # 构建系统提示词：如果智能体有自定义任务描述，优先使用
    if agent_task:
        custom_prompt = _build_agent_prompt(agent_task, web_search=web_search)
    elif web_search:
        custom_prompt = SYSTEM_PROMPT_WITH_WEB_SEARCH
    else:
        custom_prompt = SYSTEM_PROMPT
    
    if mode == "chat":
        # Chat模式：直接LLM对话，不经过Agent
        llm = create_llm(deep_think=deep_think)
        history = get_session_history(session_id)
        recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
        all_messages = recent_messages + [HumanMessage(content=user_input)]
        chat_prompt = _build_chat_prompt(agent_task) if agent_task else CHAT_SYSTEM_PROMPT
        result = llm.invoke([SystemMessage(content=chat_prompt)] + all_messages)
        full_response = result.content
        history.add_message(HumanMessage(content=user_input))
        history.add_message(AIMessage(content=full_response))
        return full_response

    agent = get_agent(web_search=web_search)

    # 获取该会话的历史消息
    history = get_session_history(session_id)

    # 只取最近 MAX_HISTORY_MESSAGES 条历史消息（加速推理）
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]

    # 构建完整的消息列表 = 历史消息 + 新消息
    all_messages = recent_messages + [HumanMessage(content=user_input)]

    # 调用 Agent
    result = agent.invoke({"messages": all_messages, "retry_count": 0})

    # 提取最后的 AI 回答
    ai_message = result["messages"][-1]

    # 保存到会话历史
    history.add_message(HumanMessage(content=user_input))
    history.add_message(ai_message)

    return ai_message.content


# ===== 5. 流式对话 =====

# 工具中文名映射
TOOL_DISPLAY_NAMES = {
    "search_documents_tool": "搜索文档",
    "lookup_employee_tool": "查询员工",
    "list_departments_tool": "部门列表",
    "list_documents_tool": "文档列表",
    "get_document_content_tool": "获取文档全文",
    "upload_document_tool": "上传文档",
    "delete_document_tool": "删除文档",
    "modify_document_tool": "修改文档",
    "export_document_tool": "导出文档",
    "web_search_tool": "联网搜索",
    "github_api_tool": "GitHub操作",        # [#12] 外部系统
    "send_email_tool": "发送邮件",          # [#12] 外部系统
    "database_query_tool": "数据库查询",    # [#12] 外部系统
}


async def chat_stream_generator(user_input: str, session_id: str = "default", web_search: bool = False, mode: str = "agent", deep_think: bool = False, agent_id: str = None, agent_task: str = None) -> AsyncGenerator[dict, None]:
    """
    流式对话：逐token输出，同时显示工具调用进度
    Yields: {"type": "token"|"tool"|"thinking"|"done"|"error", ...}
    
    [#6] 多步骤任务编排：max_tool_rounds=8 支持复杂任务链
    [#7] 工具并行执行：LangGraph 自动并行处理多个 tool_calls
    [#11] 自省纠错：should_continue 中检测工具错误并允许重试
    """
    # 设置当前智能体上下文（知识库隔离）
    set_current_agent_id(agent_id)
    # Chat模式：直接LLM对话
    if mode == "chat":
        async for chunk in _chat_mode_stream(user_input, session_id, deep_think=deep_think, web_search=web_search, agent_id=agent_id, agent_task=agent_task):
            yield chunk
        return

    # Agent模式：走Agent工具调用
    # 如果智能体有自定义任务描述，构建专属提示词
    if agent_task:
        custom_system_prompt = _build_agent_prompt(agent_task, web_search=web_search)
        agent = get_agent_with_prompt(custom_system_prompt, web_search=web_search)
    else:
        agent = get_agent(web_search=web_search)
    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
    all_messages = recent_messages + [HumanMessage(content=user_input)]

    full_response = ""
    start_time = time.time()

    try:
        # 先发一个"思考中"信号
        yield {"type": "thinking", "content": "正在思考..."}

        async for event in agent.astream_events(
            {"messages": all_messages, "retry_count": 0},
            version="v2",
        ):
            kind = event["event"]

            # LLM 输出 token（逐字输出）
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = getattr(chunk, 'content', '')
                # 处理content为列表的情况（某些LLM返回结构化内容）
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = ''.join(text_parts)
                if content and isinstance(content, str):
                    full_response += content
                    yield {"type": "token", "content": content}

            # 工具调用开始
            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                yield {"type": "tool", "name": tool_name, "display": display_name}

            # 工具调用结束
            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                yield {"type": "tool_done", "name": tool_name, "display": display_name}

    except asyncio.TimeoutError:
        yield {"type": "error", "content": "请求超时，LLM服务响应过慢，请稍后重试"}
        return
    except Exception as e:
        logger.error(f"Agent 流式输出异常: {e}", exc_info=True)
        # 流式失败 → 回退到非流式
        try:
            result = agent.invoke({"messages": all_messages, "retry_count": 0})
            ai_message = result["messages"][-1]
            full_response = ai_message.content or ""
            if full_response:
                # 模拟打字效果
                for i in range(0, len(full_response), 3):
                    yield {"type": "token", "content": full_response[i:i+3]}
                    await asyncio.sleep(0.02)
        except Exception as e2:
            yield {"type": "error", "content": f"处理失败: {str(e2)}"}
            return

    # 如果流式输出为空但Agent可能有回复（工具调用后token丢失），尝试非流式回退
    if not full_response:
        try:
            result = agent.invoke({"messages": all_messages, "retry_count": 0})
            ai_message = result["messages"][-1]
            full_response = ai_message.content or ""
            if full_response:
                for i in range(0, len(full_response), 3):
                    yield {"type": "token", "content": full_response[i:i+3]}
                    await asyncio.sleep(0.02)
            else:
                yield {"type": "error", "content": "未能获取到回复，请重试"}
        except Exception as e3:
            yield {"type": "error", "content": f"处理失败: {str(e3)}"}

    # 保存到会话历史
    if full_response:
        try:
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    # [#20] 可观测性：记录性能指标
    elapsed = time.time() - start_time
    logger.info(f"Agent 对话完成 | 耗时={elapsed:.2f}s | 模型={settings.LLM_MODEL} | 工具轮数={sum(1 for m in all_messages if isinstance(m, ToolMessage))}")

    yield {"type": "done"}


async def _chat_mode_stream(user_input: str, session_id: str = "default", deep_think: bool = False, web_search: bool = False, agent_id: str = None, agent_task: str = None) -> AsyncGenerator[dict, None]:
    """Chat模式：直接LLM流式对话，不经过Agent工具调用，可选联网搜索"""
    # Chat模式也设置智能体上下文
    set_current_agent_id(agent_id)
    # 构建系统提示词：有自定义任务描述时优先使用
    chat_system_prompt = _build_chat_prompt(agent_task) if agent_task else CHAT_SYSTEM_PROMPT
    llm = create_llm(deep_think=deep_think)
    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]
    
    # 如果开启联网搜索，先搜索再将结果注入消息
    search_context = ""
    if web_search:
        try:
            yield {"type": "thinking", "content": "正在联网搜索..."}
            yield {"type": "tool", "name": "web_search_tool", "display": "联网搜索"}
            from app.agent.tools import web_search_tool
            search_result = web_search_tool.invoke(user_input)
            yield {"type": "tool_done", "name": "web_search_tool", "display": "联网搜索"}
            search_context = f"\n\n【联网搜索结果】\n{search_result}\n\n请根据以上联网搜索结果回答用户问题。如果搜索结果没有相关信息，请根据自身知识回答。"
        except Exception as e:
            yield {"type": "tool_done", "name": "web_search_tool", "display": "联网搜索"}
            search_context = f"\n\n【联网搜索失败：{str(e)}】请根据自身知识回答。"
    
    enhanced_input = user_input + search_context
    all_messages = recent_messages + [HumanMessage(content=enhanced_input)]

    full_response = ""

    try:
        if deep_think:
            yield {"type": "thinking", "content": "深度思考中..."}
        else:
            yield {"type": "thinking", "content": "正在思考..."}

        # 添加超时控制，防止LLM长时间无响应陷入"思考循环"
        import asyncio
        chunk_count = 0
        first_token_received = False
        
        async for chunk in llm.astream([SystemMessage(content=chat_system_prompt)] + all_messages):
            content = getattr(chunk, 'content', '')
            # 处理content为列表的情况
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = ''.join(text_parts)
            if content and isinstance(content, str):
                if not first_token_received:
                    first_token_received = True
                chunk_count += 1
                full_response += content
                yield {"type": "token", "content": content}

    except asyncio.TimeoutError:
        yield {"type": "error", "content": "请求超时，LLM服务响应过慢，请稍后重试"}
        return
    except Exception as e:
        yield {"type": "error", "content": f"处理失败: {str(e)}"}
        return

    # 保存到会话历史
    if full_response:
        try:
            history.add_message(HumanMessage(content=user_input))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    yield {"type": "done"}


async def chat_stream_generator_multimodal(multimodal_content: list, session_id: str = "default") -> AsyncGenerator[dict, None]:
    """
    多模态流式对话：支持图片+文本的混合消息
    multimodal_content: [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
    注意：图片消息不走 Agent 工具调用（多模态模型不支持 function calling + 图片混用）
    """
    # 自动切换到视觉模型（当前模型不支持图片时）
    current_model = settings.LLM_MODEL
    use_model = current_model
    if current_model not in VISION_MODELS:
        use_model = DEFAULT_VISION_MODEL

    llm = ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=use_model,
        temperature=0.1,
        streaming=True,
        max_tokens=8192,
        request_timeout=60,
    )

    history = get_session_history(session_id)
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]

    # 构建多模态 HumanMessage
    human_msg = HumanMessage(content=multimodal_content)
    all_messages = recent_messages + [human_msg]

    full_response = ""

    try:
        yield {"type": "thinking", "content": f"正在分析图片（使用{use_model}）..."}

        async for chunk in llm.astream([SystemMessage(content=SYSTEM_PROMPT)] + all_messages):
            content = getattr(chunk, 'content', '')
            # 处理content为列表的情况
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = ''.join(text_parts)
            if content and isinstance(content, str):
                full_response += content
                yield {"type": "token", "content": content}

    except Exception as e:
        # 多模态失败，尝试降级为纯文本描述
        try:
            # 提取文本部分
            text_parts = [p["text"] for p in multimodal_content if p["type"] == "text"]
            fallback_text = "\n".join(text_parts) + "\n\n[注意：图片分析失败，请用文字描述你的问题]"
            async for event in chat_stream_generator(fallback_text, session_id):
                yield event
            return
        except Exception as e2:
            yield {"type": "error", "content": f"图片分析失败: {str(e2)}"}
            return

    # 保存到会话历史（用文本描述保存，避免base64占空间）
    if full_response:
        try:
            text_summary = " ".join([p["text"] for p in multimodal_content if p["type"] == "text"])
            history.add_message(HumanMessage(content=text_summary))
            history.add_message(AIMessage(content=full_response))
        except Exception:
            pass

    yield {"type": "done"}
