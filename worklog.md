
---
Task ID: 1
Agent: Main Agent
Task: 修复刷新页面后智能体列表为空的bug

Work Log:
- 分析了app.js中的自动登录流程tryAutoLogin()
- 发现tryAutoLogin()成功后缺少renderMyAgents()和updateKbUploadVisibility()调用
- 对比doLogin()手动登录流程确认了这两个调用是必需的
- 修复：在tryAutoLogin()成功后添加renderMyAgents()和updateKbUploadVisibility()调用
- 已推送到GitHub，新SHA: 050975c5a4cad09a79de63cb3f26b6af0ee277c6

Stage Summary:
- Bug根因：自动登录时只调用了loadChatList()和loadModels()，没有调用renderMyAgents()渲染智能体列表
- 修复方案：在tryAutoLogin()成功分支中补充renderMyAgents()和updateKbUploadVisibility()
- 同时确认：删除智能体时知识库同步删除的流程是正确的
- 同时确认：📊统计按钮已移除，📚知识库管理面板已存在


---
Task ID: 1+2
Agent: Main Agent
Task: 添加智能体编辑功能 + 让智能体任务描述优先于默认系统提示词

Work Log:
- app.js: 在智能体列表项的删除按钮旁添加✏️修改按钮
- app.js: 新增 openAgentEditModal/closeAgentEditModal/saveAgentEdit 函数
- app.js: 文件上传路径(chat-with-file/stream)现在也传递agent_task参数
- index.html: 添加agentEditModal弹窗（复用创建弹窗的样式）
- style.css: 添加.agent-action-btn.edit:hover样式，agentEditModal复用agentCreateModal样式
- routes.py: ChatRequest新增agent_task字段，chat-with-file/stream端点新增agent_task参数
- core.py: 新增 _build_agent_prompt() 构建智能体专属系统提示词
- core.py: 新增 _build_chat_prompt() 构建Chat模式自定义提示词
- core.py: 新增 get_agent_with_prompt() 创建自定义提示词的Agent实例
- core.py: chat()/chat_stream_generator()/_chat_mode_stream() 均接收agent_task参数
- 设计方案：agent_task有值时，用自定义角色定义替换默认「小智」角色，但保留工具使用指南和安全边界
- 修复了_chat_mode_stream中set_current_agent_id(agent_id)"的语法错误

Stage Summary:
- 智能体编辑功能完成：✏️按钮→弹窗修改名称和任务描述→保存到localStorage
- 系统提示词优先级完成：智能体任务描述 > 默认「小智」提示词
- 所有5个文件已推送到GitHub
