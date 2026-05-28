"""
FastAPI 路由定义
提供 REST API 接口供外部调用
包含：认证（JWT）、聊天（含流式）、文档管理、会话管理、模型管理、统计

优化:
- [#20] 可观测性：请求日志中间件 + 性能指标
- [#22] 配置中心：运行时热更新配置 API
- [#23] API 分页：对话列表/文档列表支持分页
- [#24] 健康检查增强：检查 ChromaDB/LLM API/磁盘等依赖
"""
import os
import asyncio
import time
import shutil
import json
import base64
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from app.agent.core import chat, chat_stream_generator, chat_stream_generator_multimodal, reset_agent
from app.agent.storage import sync_agents as sync_agents_storage, load_agents, debug_info as agent_debug_info
from app.rag.document import index_document, search_documents, list_indexed_documents, delete_document, update_document, delete_agent_collection, list_all_collections, load_document, export_document_as_docx, reindex_all_documents
from app.auth.user_manager import login_user, register_user
from app.auth.jwt_handler import create_token, verify_token, get_username_from_token
from app.memory.manager import (
    get_history_messages, clear_session_history,
    create_chat, list_chats, delete_chat, rename_chat, update_chat_time,
)
from app.config import settings, AVAILABLE_MODELS, get_current_model, set_current_model
from app.utils.stats import record_message, record_session, get_stats

logger = logging.getLogger(__name__)

# 文件大小限制：50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

router = APIRouter()


# ===== [#20] 可观测性：请求计时 + 性能日志 =====
_request_stats = {
    "total_requests": 0,
    "total_errors": 0,
    "avg_response_time": 0.0,
    "endpoint_stats": {},  # path -> {count, avg_time, errors}
}


def _record_request(path: str, duration: float, is_error: bool = False):
    """记录请求统计"""
    _request_stats["total_requests"] += 1
    if is_error:
        _request_stats["total_errors"] += 1
    
    # 更新平均响应时间
    total = _request_stats["total_requests"]
    prev_avg = _request_stats["avg_response_time"]
    _request_stats["avg_response_time"] = prev_avg + (duration - prev_avg) / total
    
    # 端点统计
    if path not in _request_stats["endpoint_stats"]:
        _request_stats["endpoint_stats"][path] = {"count": 0, "avg_time": 0.0, "errors": 0}
    ep = _request_stats["endpoint_stats"][path]
    ep["count"] += 1
    prev = ep["avg_time"]
    ep["avg_time"] = prev + (duration - prev) / ep["count"]
    if is_error:
        ep["errors"] += 1


# ===== JWT 认证依赖 =====
def get_current_user(request: Request) -> str:
    """
    从请求中提取当前用户名（JWT Token 或兼容旧方式）
    不强制认证，但如果有 Token 则验证
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        username = get_username_from_token(token)
        if username:
            return username
    # 兼容：从查询参数获取
    username = request.query_params.get("username", "")
    return username


def require_auth(request: Request) -> str:
    """
    强制要求 JWT 认证
    返回已认证的用户名
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        username = get_username_from_token(token)
        if username:
            return username
    raise HTTPException(status_code=401, detail="未认证，请重新登录")


# ===== 请求/响应模型 =====
class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: str = "default"
    web_search: bool = False
    mode: str = "agent"  # agent / chat
    deep_think: bool = False
    agent_id: str = None  # 智能体ID，用于知识库隔离
    agent_task: str = None  # 智能体任务描述，用于动态系统提示词


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    session_id: str


class SearchRequest(BaseModel):
    """文档搜索请求"""
    query: str
    top_k: int = 3


class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """注册请求"""
    username: str
    password: str


class ModelSetRequest(BaseModel):
    """设置模型请求"""
    model_id: str


class RenameRequest(BaseModel):
    """重命名会话请求"""
    username: str
    chat_id: str
    new_title: str


# [#22] 配置中心请求模型
class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    key: str  # 配置项名称，如 LLM_MODEL, MAX_TOOL_ROUNDS 等
    value: str  # 新值（字符串形式，内部转换）


class ModifyDocumentRequest(BaseModel):
    """修改知识库文档请求"""
    content: str  # 新的文档内容（纯文本）
    append: bool = False  # 是否追加内容（True=在原文末尾追加，False=替换全部内容）
    return_docx: bool = False  # 是否同时返回修改后的docx文件下载链接


class ExportDocumentRequest(BaseModel):
    """导出/生成文档请求"""
    content: str  # 文档内容（纯文本）
    filename: str = ""  # 输出文件名（含扩展名），为空则自动生成
    title: str = ""  # 文档标题，为空则使用filename

class AgentSyncRequest(BaseModel):
    """智能体同步请求"""
    agents: list  # 智能体列表



# ===== 认证接口 =====

@router.post("/auth/login", summary="用户登录")
async def auth_login(req: LoginRequest):
    """用户登录验证，返回 JWT Token"""
    start = time.time()
    try:
        result = login_user(req.username, req.password)
        if result.get("success"):
            # 签发 JWT Token
            token = create_token(req.username)
            result["token"] = token
        return result
    finally:
        _record_request("/auth/login", time.time() - start)


@router.post("/auth/register", summary="用户注册")
async def auth_register(req: RegisterRequest):
    """用户注册"""
    start = time.time()
    try:
        result = register_user(req.username, req.password)
        if result.get("success"):
            # 注册成功也签发 Token
            token = create_token(req.username)
            result["token"] = token
        return result
    finally:
        _record_request("/auth/register", time.time() - start)


@router.get("/auth/me", summary="验证 Token 有效性")
async def auth_me(request: Request):
    """验证当前 JWT Token 是否有效"""
    try:
        username = require_auth(request)
        return {"valid": True, "username": username}
    except HTTPException:
        return {"valid": False, "username": None}


# ===== 聊天接口 =====

@router.post("/chat", response_model=ChatResponse, summary="与 Agent 对话（非流式）")
async def chat_api(req: ChatRequest, username: str = Depends(get_current_user)):
    """
    核心接口：与文档助手 Agent 对话（非流式）

    - 支持 RAG 文档问答
    - 支持员工信息查询
    - 支持多轮对话
    """
    start = time.time()
    try:
        response = chat(req.message, req.session_id, web_search=req.web_search, mode=req.mode, deep_think=req.deep_think, agent_id=req.agent_id, agent_task=req.agent_task)
        # 更新会话时间
        try:
            parts = req.session_id.split("_", 1)
            if len(parts) == 2:
                update_chat_time(parts[0], req.session_id)
        except Exception:
            pass
        # 记录统计
        record_message(username=username or "anonymous", model_id=get_current_model())
        return ChatResponse(response=response, session_id=req.session_id)
    except Exception as e:
        _record_request("/chat", time.time() - start, is_error=True)
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")
    finally:
        _record_request("/chat", time.time() - start)


@router.post("/chat/stream", summary="与 Agent 对话（流式 SSE）")
async def chat_stream_api(req: ChatRequest, username: str = Depends(get_current_user)):
    """
    流式对话接口：逐 token 输出，同时显示工具调用进度
    返回 Server-Sent Events (SSE) 流
    """
    start = time.time()
    # 记录统计
    record_message(username=username or "anonymous", model_id=get_current_model())

    async def event_generator():
        async for chunk in chat_stream_generator(req.message, req.session_id, web_search=req.web_search, mode=req.mode, deep_think=req.deep_think, agent_id=req.agent_id, agent_task=req.agent_task):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        # 更新会话时间
        try:
            parts = req.session_id.split("_", 1)
            if len(parts) == 2:
                update_chat_time(parts[0], req.session_id)
        except Exception:
            pass
        _record_request("/chat/stream", time.time() - start)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat-with-file/stream", summary="带文件的流式对话")
async def chat_with_file_stream(
    file: UploadFile = File(...),
    message: str = Form(""),
    session_id: str = Form("default"),
    web_search: bool = Form(False),
    mode: str = Form("agent"),
    deep_think: bool = Form(False),
    agent_id: str = Form(None),
    agent_task: str = Form(None),
    store_to_kb: str = Form("true"),
    username: str = Depends(get_current_user),
):
    """
    带文件的流式对话：支持图片和文档
    - 图片（png/jpg/jpeg/gif/bmp/webp）：转为base64传给LLM分析
    - 文档（pdf/txt/docx）：索引后基于内容回答
    - 其他文件：读取文本内容（如有）传给LLM
    返回 Server-Sent Events (SSE) 流
    """
    start = time.time()
    # 记录统计
    record_message(username=username or "anonymous", model_id=get_current_model())

    ext = os.path.splitext(file.filename)[1].lower()
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    doc_exts = {".pdf", ".txt", ".docx"}
    code_exts = {".py", ".js", ".html", ".css", ".json", ".md", ".csv", ".xlsx", ".xls", ".doc", ".ppt", ".pptx"}

    # 文件大小检查
    file_content_raw = await file.read()
    if len(file_content_raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 50MB），当前文件: {len(file_content_raw) // 1024 // 1024}MB")
    # 重置文件指针
    await file.seek(0)

    logger.info(f"收到文件上传: {file.filename}, 大小: {len(file_content_raw)} bytes")

    if ext in image_exts:
        # 图片文件：用多模态消息格式传给LLM做视觉分析
        file_content = await file.read()
        b64 = base64.b64encode(file_content).decode("utf-8")
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")
        # 构建多模态消息内容
        image_url = f"data:{mime_type};base64,{b64}"
        multimodal_content = [
            {"type": "text", "text": f"[用户上传了图片: {file.filename}]\n\n{message or '请描述这张图片'}"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        # 直接调用多模态流式生成
        async def event_generator():
            async for chunk in chat_stream_generator_multimodal(multimodal_content, session_id):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            try:
                parts = session_id.split("_", 1)
                if len(parts) == 2:
                    update_chat_time(parts[0], session_id)
            except Exception:
                pass
            _record_request("/chat-with-file/stream", time.time() - start)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    elif ext in doc_exts:
        # 文档文件
        file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        if store_to_kb == "true":
            # 知识库模式 ON：索引到知识库后回答
            try:
                # 空字符串转为 None
                aid = agent_id if agent_id else None
                index_result = index_document(file_path, file.filename, agent_id=aid)
                print(f"[DEBUG-上传] 文件已索引到知识库: {file.filename}, agent_id={aid}, 分块数={index_result.get('chunks', 0)}")
            except Exception as e:
                os.remove(file_path)
                raise HTTPException(status_code=500, detail=f"文档索引失败: {str(e)}")
            full_message = f"[用户上传了文档: {file.filename}]\n\n{message}"
        else:
            # 知识库模式 OFF：只读取内容回答，不存入知识库
            try:
                docs = load_document(file_path)
                text = "\n".join([doc.page_content for doc in docs])
                full_message = f"[用户上传了文档: {file.filename}]\n\n文档内容：\n{text[:8000]}\n\n{message}"
                print(f"[DEBUG-上传] 文件仅读取内容（不存知识库）: {file.filename}")
            except Exception as e:
                os.remove(file_path)
                raise HTTPException(status_code=500, detail=f"文档读取失败: {str(e)}")

    elif ext in code_exts:
        # 代码/其他文本文件：读取内容传给LLM
        try:
            file_content = await file.read()
            text = file_content.decode("utf-8", errors="replace")
            full_message = f"[用户上传了文件: {file.filename}]\n\n文件内容：\n```\n{text[:8000]}\n```\n\n{message}"
        except Exception:
            full_message = f"[用户上传了文件: {file.filename}，但无法读取内容]\n\n{message}"
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    # 流式回答
    async def event_generator():
        aid = agent_id if agent_id else None
        atask = agent_task if agent_task else None
        async for chunk in chat_stream_generator(full_message, session_id, web_search=web_search, mode=mode, deep_think=deep_think, agent_id=aid, agent_task=atask):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        try:
            parts = session_id.split("_", 1)
            if len(parts) == 2:
                update_chat_time(parts[0], session_id)
        except Exception:
            pass
        _record_request("/chat-with-file/stream", time.time() - start)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===== 文档管理接口 =====

@router.post("/upload", summary="上传文档到知识库")
async def upload_document(file: UploadFile = File(...), agent_id: str = Form(None)):
    """
    上传文档并自动索引到向量数据库
    支持 PDF、TXT、DOCX 格式
    agent_id: 智能体ID，为空时索引到全局知识库
    """
    # 检查文件格式
    allowed_ext = {".pdf", ".txt", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，仅支持 {allowed_ext}",
        )

    # 文件大小检查
    file_content_raw = await file.read()
    if len(file_content_raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 50MB）")
    await file.seek(0)

    logger.info(f"知识库上传文档: {file.filename}, 大小: {len(file_content_raw)} bytes")

    # 保存文件
    file_path = os.path.join(settings.DOCUMENTS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 索引文档
    try:
        result = index_document(file_path, file.filename, agent_id=agent_id)
        return {"status": "success", "detail": result}
    except Exception as e:
        # 索引失败则删除文件
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档索引失败: {str(e)}")


@router.post("/search", summary="搜索文档内容")
async def search_api(req: SearchRequest):
    """在文档库中搜索相关内容"""
    results = search_documents(req.query, req.top_k)
    return {"query": req.query, "results": results}


@router.get("/documents", summary="列出所有已索引文档")
async def list_documents(
    page: int = Query(1, ge=1, description="页码"),          # [#23] 分页
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    agent_id: str = Query(None, description="智能体ID，为空时查全局知识库"),
):
    """获取知识库中所有文档列表（支持分页，按智能体隔离）"""
    docs = list_indexed_documents(agent_id=agent_id)

    # BUG FIX: Filesystem fallback - scan DOCUMENTS_DIR for files not in ChromaDB
    # When ChromaDB has no records (e.g. embedding unavailable), files on disk should still appear
    doc_extensions = {'.pdf', '.txt', '.docx', '.csv', '.xlsx', '.xls', '.doc', '.ppt', '.pptx', '.md', '.py', '.js', '.html', '.css', '.json'}
    chromadb_filenames = set()
    for doc in docs:
        if isinstance(doc, dict) and doc.get('filename'):
            chromadb_filenames.add(doc['filename'])
        elif isinstance(doc, str):
            chromadb_filenames.add(doc)

    if os.path.exists(settings.DOCUMENTS_DIR):
        for fname in os.listdir(settings.DOCUMENTS_DIR):
            ext = os.path.splitext(fname)[1].lower()
            if ext in doc_extensions and fname not in chromadb_filenames:
                file_path = os.path.join(settings.DOCUMENTS_DIR, fname)
                if os.path.isfile(file_path):
                    try:
                        file_stat = os.stat(file_path)
                        docs.append({
                            'filename': fname,
                            'source': 'filesystem',
                            'size': file_stat.st_size,
                            'modified': file_stat.st_mtime,
                        })
                    except OSError:
                        pass

    total = len(docs)
    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    paginated = docs[start:end]
    return {
        "status": "success",
        "documents": paginated,
        "total": total,
        "count": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.put("/documents/{filename}", summary="修改知识库文档内容")
async def modify_document_api(filename: str, req: ModifyDocumentRequest):
    """
    修改知识库中指定文档的内容
    支持两种模式：
    - 替换模式（append=false）：用新内容完全替换原文档内容
    - 追加模式（append=true）：在原文档内容末尾追加新内容
    修改后会自动重新索引到向量数据库
    """
    # 检查文档是否存在
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")

    # 追加模式：先读取原内容，拼接新内容
    final_content = req.content
    if req.append:
        try:
            from app.rag.document import load_document
            docs = load_document(file_path)
            original_text = "\n".join([doc.page_content for doc in docs])
            final_content = original_text + "\n" + req.content
        except Exception as e:
            logger.warning(f"读取原文档内容失败，改为替换模式: {e}")

    logger.info(f"知识库修改文档: {filename}, 追加模式={req.append}, 内容长度={len(final_content)}")

    result = update_document(filename, final_content, async_reindex=True)  # 异步重索引，加速响应
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=result["message"])
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    response_data = {"status": "success", "detail": result}

    # 如果用户要求返回docx文件下载链接
    if req.return_docx:
        try:
            docx_filename = filename.rsplit('.', 1)[0] + '.docx'
            docx_result = export_document_as_docx(final_content, docx_filename)
            if docx_result["status"] == "success":
                response_data["download_url"] = f"/api/v1/documents/{docx_filename}/download"
                response_data["docx_filename"] = docx_filename
        except Exception as e:
            logger.warning(f"生成docx下载文件失败: {e}")

    return response_data


@router.get("/documents/{filename}/download", summary="下载知识库文档")
async def download_document(filename: str):
    """
    下载知识库中的文档文件
    支持 .docx / .txt / .pdf 格式
    """
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")

    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
    }
    media_type = mime_map.get(ext, "application/octet-stream")

    with open(file_path, "rb") as f:
        content = f.read()

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{filename}"
        }
    )


@router.post("/documents/export", summary="导出/生成文档为docx")
async def export_document_api(req: ExportDocumentRequest):
    """
    将文本内容生成为docx文档并提供下载
    支持从知识库内容整合生成综合文档或简略文档
    """
    try:
        filename = req.filename or f"export_{int(time.time())}.docx"
        if not filename.endswith('.docx'):
            filename += '.docx'

        result = export_document_as_docx(req.content, filename, title=req.title)
        if result["status"] == "success":
            return {
                "status": "success",
                "filename": filename,
                "download_url": f"/api/v1/documents/{filename}/download",
                "message": result["message"],
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "导出失败"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档导出失败: {str(e)}")


@router.delete("/documents/{filename}", summary="从知识库删除文档")
async def delete_document_api(filename: str, agent_id: str = Query(None, description="智能体ID，为空时删全局知识库文档")):
    """
    从知识库中删除指定文档
    同时删除 ChromaDB 中的向量分块和原始文件
    """
    result = delete_document(filename, agent_id=agent_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=result["message"])
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return {"status": "success", "detail": result}


# ===== 会话历史接口 =====

@router.get("/history/{session_id}", summary="获取对话历史")
async def get_history(session_id: str):
    """获取指定会话的对话历史"""
    messages = get_history_messages(session_id)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@router.delete("/history/{session_id}", summary="清除对话历史")
async def delete_history(session_id: str):
    """清除指定会话的对话历史"""
    clear_session_history(session_id)
    return {"status": "success", "message": f"会话 {session_id} 的历史已清除"}


# ===== 会话管理接口 =====

@router.get("/chats", summary="获取用户会话列表")
async def get_chats(
    username: str,
    mode: str = Query(None, description="模式过滤: agent/chat"),
    page: int = Query(1, ge=1, description="页码"),          # [#23] 分页
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
):
    """获取用户的会话列表（支持分页，支持按模式过滤）"""
    chats = list_chats(username, mode=mode)
    total = len(chats)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = chats[start:end]
    return {
        "success": True,
        "chats": paginated,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post("/chats", summary="创建新会话")
async def create_chat_api(username: str, title: str = "新对话", mode: str = "agent", agent_id: str = None):
    """为用户创建一个新的会话（支持指定模式和智能体关联）"""
    chat_info = create_chat(username, title, mode=mode, agent_id=agent_id)
    record_session()
    return {"success": True, "chat": chat_info}


@router.delete("/chats/{chat_id}", summary="删除会话")
async def delete_chat_api(chat_id: str, username: str):
    """删除用户的某个会话"""
    delete_chat(username, chat_id)
    return {"success": True, "message": "会话已删除"}


@router.put("/chats/{chat_id}/rename", summary="重命名会话")
async def rename_chat_api(chat_id: str, req: RenameRequest):
    """重命名用户的某个会话"""
    rename_chat(req.username, req.chat_id, req.new_title)
    return {"success": True, "message": "会话已重命名"}


# ===== 模型管理接口 =====

@router.get("/models", summary="获取可用模型列表")
async def get_models():
    """获取所有可用的 LLM 模型列表"""
    current = get_current_model()
    return {"models": AVAILABLE_MODELS, "current": current}


@router.post("/models/set", summary="切换模型")
async def set_model(req: ModelSetRequest):
    """切换当前使用的 LLM 模型"""
    success = set_current_model(req.model_id)
    if success:
        return {"success": True, "message": f"已切换到模型: {req.model_id}"}
    return {"success": False, "message": f"不支持的模型: {req.model_id}"}


# ===== 使用统计接口 =====

@router.get("/stats", summary="获取使用统计")
async def get_usage_stats(username: str = Depends(get_current_user)):
    """获取系统使用统计数据"""
    stats = get_stats()
    # [#20] 附加 API 性能指标
    stats["api_performance"] = {
        "total_requests": _request_stats["total_requests"],
        "total_errors": _request_stats["total_errors"],
        "avg_response_time_ms": round(_request_stats["avg_response_time"] * 1000, 2),
        "error_rate": round(_request_stats["total_errors"] / max(_request_stats["total_requests"], 1) * 100, 2),
    }
    return {"success": True, "stats": stats}


# ===== [#22] 配置中心 API =====

@router.get("/config", summary="获取运行时配置")
async def get_config(username: str = Depends(require_auth)):
    """获取当前运行时配置（隐藏敏感信息）"""
    return {
        "success": True,
        "config": {
            "LLM_MODEL": settings.LLM_MODEL,
            "LLM_BASE_URL": settings.LLM_BASE_URL,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
            "APP_HOST": settings.APP_HOST,
            "APP_PORT": settings.APP_PORT,
            "GITHUB_TOKEN_CONFIGURED": bool(os.getenv("GITHUB_TOKEN", "")),
            "SMTP_CONFIGURED": bool(os.getenv("SMTP_HOST", "")),
            "DATABASE_CONFIGURED": bool(os.getenv("DATABASE_URL", "")),
        }
    }


@router.post("/config", summary="更新运行时配置（热更新）")
async def update_config(req: ConfigUpdateRequest, username: str = Depends(require_auth)):
    """
    [#22] 运行时热更新配置，无需重启服务
    支持更新的配置项：LLM_MODEL, APP_PORT 等
    """
    allowed_keys = {"LLM_MODEL", "APP_PORT", "EMBEDDING_MODEL"}
    
    if req.key not in allowed_keys:
        raise HTTPException(status_code=400, detail=f"不允许更新的配置项: {req.key}。支持: {allowed_keys}")
    
    old_value = getattr(settings, req.key, None)
    if old_value is None:
        raise HTTPException(status_code=400, detail=f"未知的配置项: {req.key}")
    
    # 类型转换
    try:
        if req.key == "APP_PORT":
            new_value = int(req.value)
        else:
            new_value = req.value
    except ValueError:
        raise HTTPException(status_code=400, detail=f"配置值类型错误: {req.key} 期望 {type(old_value).__name__}")
    
    # 应用更新
    setattr(settings, req.key, new_value)
    
    # 如果更新了模型，重置 Agent
    if req.key == "LLM_MODEL":
        reset_agent()
        logger.info(f"配置热更新: {req.key} = {new_value}, Agent 已重置")
    elif req.key == "EMBEDDING_MODEL":
        from app.rag.document import reset_vector_store
        reset_vector_store()
        logger.info(f"配置热更新: {req.key} = {new_value}, 向量数据库已重置")
    
    logger.info(f"配置热更新: {req.key} 由 {old_value} 变更为 {new_value}, 操作者: {username}")
    
    return {
        "success": True,
        "message": f"配置 {req.key} 已更新",
        "old_value": str(old_value),
        "new_value": str(new_value),
    }


# ===== 导出对话接口 =====

@router.get("/export/{session_id}", summary="导出对话")
async def export_chat(session_id: str, format: str = "md"):
    """
    导出对话为 Markdown 或 PDF 格式
    format: md | pdf
    """
    messages = get_history_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="没有可导出的对话内容")

    if format == "pdf":
        # PDF 导出
        try:
            from app.utils.pdf_generator import generate_chat_pdf
            pdf_bytes = generate_chat_pdf(messages, session_id)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=chat_{session_id[:12]}.pdf"
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF 生成失败: {str(e)}")
    else:
        # Markdown 导出
        content = ""
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            content += f"**{role}：**\n\n{msg['content']}\n\n---\n\n"

        return Response(
            content=content.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=chat_{session_id[:12]}.md"
            }
        )


# ===== [#24] 健康检查增强 =====

@router.get("/health/detailed", summary="详细健康检查")
async def health_detailed():
    """
    [#24] 详细健康检查：检查所有依赖组件状态
    - ChromaDB 可用性
    - LLM API 可达性
    - 磁盘空间
    - 内存使用
    """
    import platform
    
    checks = {}
    overall = "healthy"
    
    # 1. ChromaDB 检查
    try:
        from app.rag.document import get_vector_store
        vs = get_vector_store()
        collection = vs._collection
        count = collection.count()
        checks["chromadb"] = {"status": "ok", "document_count": count}
    except Exception as e:
        checks["chromadb"] = {"status": "error", "message": str(e)[:200]}
        overall = "degraded"
    
    # 2. LLM API 检查
    try:
        import httpx
        api_url = settings.LLM_BASE_URL.rstrip("/") + "/models"
        resp = httpx.get(api_url, timeout=5)
        if resp.status_code == 200:
            checks["llm_api"] = {"status": "ok", "model": settings.LLM_MODEL}
        else:
            checks["llm_api"] = {"status": "error", "code": resp.status_code}
            overall = "degraded"
    except Exception as e:
        checks["llm_api"] = {"status": "unreachable", "message": str(e)[:100]}
        overall = "degraded"
    
    # 3. 磁盘空间检查
    try:
        disk_usage = shutil.disk_usage(settings.DATA_DIR)
        free_gb = disk_usage.free / (1024 ** 3)
        total_gb = disk_usage.total / (1024 ** 3)
        usage_pct = (disk_usage.used / disk_usage.total) * 100
        checks["disk"] = {
            "status": "ok" if usage_pct < 90 else "warning",
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "usage_percent": round(usage_pct, 1),
        }
        if usage_pct >= 90:
            overall = "degraded"
    except Exception as e:
        checks["disk"] = {"status": "error", "message": str(e)[:100]}
    
    # 4. 内存检查
    try:
        import psutil
        mem = psutil.virtual_memory()
        checks["memory"] = {
            "status": "ok" if mem.percent < 90 else "warning",
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "used_percent": mem.percent,
        }
    except ImportError:
        checks["memory"] = {"status": "unknown", "message": "psutil not installed"}
    
    # 5. 系统信息
    checks["system"] = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "version": "4.0.0",
    }
    
    return {
        "status": overall,
        "checks": checks,
        "timestamp": time.time(),
    }


# ===== 智能体知识库管理接口 =====

@router.delete("/agents/{agent_id}/knowledge", summary="删除智能体的知识库")
async def delete_agent_knowledge(agent_id: str):
    """
    删除智能体对应的整个 ChromaDB collection
    在删除智能体时调用，确保知识库数据同步清理
    """
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id 不能为空")
    result = delete_agent_collection(agent_id)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return {"status": "success", "detail": result}


# ===== 诊断接口 =====

@router.get("/debug/collections", summary="列出所有 ChromaDB collection")
async def debug_collections():
    """诊断接口：列出所有 ChromaDB collection 及其文档数"""
    collections = list_all_collections()
    return {"collections": collections}





# ===== 智能体同步接口 =====

@router.post("/agents/sync", summary="同步智能体数据（跨浏览器/设备同步）")
async def sync_agents_api(
    req: AgentSyncRequest,
    username: str = Depends(require_auth),
):
    """
    同步智能体数据：
    - 客户端上传本地智能体列表
    - 服务端按 agent_id 合并（不覆盖，取较新版本）
    - 返回合并后的完整列表
    
    解决跨浏览器/设备智能体数据不一致问题
    """
    if not username:
        raise HTTPException(status_code=401, detail="未认证，请重新登录")
    
    try:
        result = sync_agents_storage(username, req.agents)
        # 过滤：只保留允许的智能体ID
        allowed_ids = {'xf-rd-agent', 'xf-quality-agent'}
        filtered_agents = [a for a in result["agents"] if a.get("id") in allowed_ids]
        return {
            "success": True,
            "agents": filtered_agents,
            "synced": result["synced"],
            "added": result["added"],
            "updated": result["updated"],
            "total": len(filtered_agents),
        }
    except Exception as e:
        logger.error(f"智能体同步失败 [{username}]: {e}")
        raise HTTPException(status_code=500, detail=f"同步失败: {str(e)}")


@router.get("/agents", summary="获取用户的智能体列表")
async def get_agents(username: str = Depends(require_auth)):
    """
    获取当前用户的所有智能体
    """
    if not username:
        raise HTTPException(status_code=401, detail="未认证，请重新登录")
    
    agents = load_agents(username)
    # 过滤：只保留允许的智能体ID
    allowed_ids = {'xf-rd-agent', 'xf-quality-agent'}
    agents = [a for a in agents if a.get("id") in allowed_ids]
    return {
        "success": True,
        "agents": agents,
        "total": len(agents),
    }


@router.get("/agents/debug", summary="智能体数据诊断")
async def agents_debug(username: str = Depends(require_auth)):
    """
    诊断接口：返回用户智能体数据的详细信息
    用于排查跨浏览器同步问题
    """
    if not username:
        raise HTTPException(status_code=401, detail="未认证，请重新登录")
    
    info = agent_debug_info(username)
    return {
        "success": True,
        "debug": info,
    }


@router.post("/reindex", summary="重建知识库索引（切换embedding模型后使用）")
async def reindex_knowledge(agent_id: str = Query(None, description="智能体ID，为空时重建全局知识库")):
    """
    重建指定知识库的所有文档索引。
    
    切换embedding模型后（如从智谱embedding-3切换到本地bge-large-zh-v1.5），
    旧的向量数据维度不同，必须重建索引才能正常使用向量搜索。
    
    此接口会：
    1. 记录旧collection中的文档列表
    2. 删除旧collection
    3. 用新的embedding模型重新索引所有文档
    """
    result = reindex_all_documents(agent_id=agent_id)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return {"status": "success", "detail": result}


@router.get("/migrate/cleanup-collections", summary="清理异常的 ChromaDB collection")
async def cleanup_collections():
    """
    清理空 collection 或有双重前缀的 collection
    例如：agent_agent_xxx → 应该是 agent_xxx
    """
    import chromadb
    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    collections = client.list_collections()
    cleaned = []

    for c in collections:
        name = c.name
        # 修复双重前缀：agent_agent_xxx → agent_xxx
        if name.startswith("agent_agent_"):
            correct_name = name.replace("agent_agent_", "agent_", 1)
            try:
                # 获取旧 collection 的数据
                old_data = c.get(include=["documents", "metadatas", "embeddings"])
                if old_data.get("ids"):
                    # 创建正确名称的 collection 并迁移数据
                    from app.rag.document import get_vector_store
                    # 从 agent_agent_xxx 提取真正的 agent_id
                    real_agent_id = name.replace("agent_", "", 1)  # 去掉第一个 agent_ 前缀
                    new_vs = get_vector_store(agent_id=real_agent_id)
                    # 迁移文档
                    from langchain_core.documents import Document
                    docs = []
                    for i, doc_id in enumerate(old_data["ids"]):
                        doc = Document(
                            page_content=old_data["documents"][i] or "",
                            metadata=old_data["metadatas"][i] or {},
                        )
                        docs.append(doc)
                    if docs:
                        new_vs.add_documents(docs)
                    cleaned.append({"old": name, "new": correct_name, "migrated_docs": len(docs)})
                # 删除旧 collection
                client.delete_collection(name)
            except Exception as e:
                cleaned.append({"old": name, "error": str(e)})
        # 清理空 collection（除了 langchain）
        elif name != "langchain":
            try:
                count = c.count()
                if count == 0:
                    client.delete_collection(name)
                    cleaned.append({"deleted_empty": name})
            except:
                pass

    # 清理 vector_store 缓存
    from app.rag.document import reset_vector_store
    reset_vector_store()

    return {"status": "success", "cleaned": cleaned}
