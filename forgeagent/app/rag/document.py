"""
文档处理与向量化模块 (RAG)
负责：加载文档 → 分块 → 向量化 → 存入 ChromaDB → 检索

优化:
- [#9] RAG 检索质量提升：混合检索（向量 + BM25关键词） + 重排序
- [#10] 引用溯源：返回结果标注文档名 + 段落位置 + chunk_id
"""
import os
import re
import json
import logging
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from app.config import settings

logger = logging.getLogger(__name__)

# 本地 Embedding 批量大小（本地模型无API限制，可适当增大）
EMBEDDING_BATCH_SIZE = 100

# ===== 单例模式：复用 Embedding 和 ChromaDB 连接 =====
_embeddings_instance = None
_vector_store_cache = {}  # agent_id -> ChromaDB instance（按智能体隔离）

# 全局知识库的 collection 名称
GLOBAL_COLLECTION_NAME = "langchain"

# Embedding 提供者配置："local" 使用本地免费模型，"openai" 使用云端API（消耗额度）
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")


def get_embeddings():
    """获取 Embedding 模型（单例复用，避免重复初始化）
    
    支持两种模式：
    - local（默认）：使用本地 BAAI/bge-large-zh-v1.5，永久免费，离线可用，中文质量顶级
    - openai：使用智谱 embedding-3 云端API，消耗额度，需联网
    
    通过环境变量 EMBEDDING_PROVIDER 切换：local / openai
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        provider = os.environ.get("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER)
        
        if provider == "local":
            # ===== 本地 Embedding（免费，离线，中文顶级）=====
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
                logger.info(f"正在加载本地 Embedding 模型: {model_name}（首次运行会自动下载，约1.2GB）...")
                _embeddings_instance = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info(f"本地 Embedding 模型加载完成: {model_name}")
            except ImportError:
                logger.error("langchain-huggingface 未安装，请运行: pip install langchain-huggingface sentence-transformers")
                logger.info("回退到 OpenAI Embedding...")
                provider = "openai"
            except Exception as e:
                logger.error(f"本地 Embedding 加载失败: {e}")
                logger.info("回退到 OpenAI Embedding...")
                provider = "openai"
        
        if provider == "openai":
            # ===== 云端 API Embedding（消耗额度，需联网）=====
            from langchain_openai import OpenAIEmbeddings
            embedding_model = getattr(settings, 'EMBEDDING_MODEL', 'embedding-3')
            _embeddings_instance = OpenAIEmbeddings(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
                model=embedding_model,
            )
            logger.info(f"Embedding 模型已初始化（云端API）: {embedding_model}")
    
    return _embeddings_instance


def _get_collection_name(agent_id: str = None) -> str:
    """根据 agent_id 获取 ChromaDB collection 名称
    
    - agent_id 为 None 或空 → 全局知识库（普通Agent模式）
    - agent_id 有值 → 智能体专属知识库
    """
    if agent_id:
        # 用 agent_id 做 collection 名，确保合法
        safe_id = agent_id.replace('-', '_').replace(' ', '_')
        return f"agent_{safe_id}"
    return GLOBAL_COLLECTION_NAME


def get_vector_store(agent_id: str = None):
    """获取 ChromaDB 向量数据库实例（按 agent_id 隔离）
    
    Args:
        agent_id: 智能体ID，为 None 时使用全局知识库
    
    每个智能体有独立的 ChromaDB collection，互不干扰。
    普通 Agent 模式使用默认的全局 collection。
    """
    cache_key = agent_id or "__global__"
    
    if cache_key not in _vector_store_cache:
        embeddings = get_embeddings()
        collection_name = _get_collection_name(agent_id)
        vs = Chroma(
            collection_name=collection_name,
            persist_directory=settings.CHROMA_DIR,
            embedding_function=embeddings,
        )
        _vector_store_cache[cache_key] = vs
        logger.info(f"ChromaDB 已连接: collection={collection_name}, agent_id={agent_id}")
    
    return _vector_store_cache[cache_key]


def reset_vector_store():
    """重置向量数据库单例（配置变更时调用）"""
    global _embeddings_instance
    _vector_store_cache.clear()
    _embeddings_instance = None
    logger.info("向量数据库单例已重置")


def reindex_all_documents(agent_id: str = None):
    """重建指定知识库的所有文档索引（切换embedding模型后调用）
    
    当从 OpenAI Embedding 切换到本地 Embedding 时，
    旧向量数据的维度不同，需要删除旧collection并重新索引。
    
    Args:
        agent_id: 智能体ID，为None时重建全局知识库
    
    Returns:
        dict: 包含重建结果
    """
    import chromadb
    collection_name = _get_collection_name(agent_id)
    
    try:
        # 1. 获取旧collection中的所有文档文件名
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        existing_collections = [c.name for c in client.list_collections()]
        
        document_files = []
        if collection_name in existing_collections:
            collection = client.get_collection(collection_name)
            all_docs = collection.get(include=["metadatas"])
            for meta in (all_docs.get("metadatas") or []):
                if meta and "source_file" in meta:
                    document_files.append(meta["source_file"])
            document_files = list(set(document_files))
            
            # 2. 删除旧collection
            client.delete_collection(collection_name)
            logger.info(f"已删除旧collection: {collection_name}")
        
        # 3. 清除缓存，让新的embedding生效
        cache_key = agent_id or "__global__"
        if cache_key in _vector_store_cache:
            del _vector_store_cache[cache_key]
        
        # 4. 重新索引所有文档
        reindexed = []
        failed = []
        for filename in document_files:
            file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
            if os.path.exists(file_path):
                try:
                    result = index_document(file_path, filename, agent_id=agent_id)
                    reindexed.append(filename)
                    logger.info(f"重新索引成功: {filename}, {result.get('chunks', 0)} 个分块")
                except Exception as e:
                    failed.append(f"{filename}: {str(e)}")
                    logger.error(f"重新索引失败: {filename}, {e}")
            else:
                failed.append(f"{filename}: 文件不存在")
        
        return {
            "status": "success",
            "collection": collection_name,
            "documents_found": len(document_files),
            "reindexed": len(reindexed),
            "failed": failed,
            "message": f"知识库重建完成: 找到{len(document_files)}个文档，成功索引{len(reindexed)}个" + (f"，失败{len(failed)}个" if failed else "")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"重建知识库失败: {str(e)}"
        }


def load_document(file_path: str) -> list:
    """
    根据文件类型加载文档
    支持：PDF、TXT、DOCX
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 PDF/TXT/DOCX")

    return loader.load()


def split_documents(docs: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    文档分块
    - chunk_size: 每块最大字符数
    - chunk_overlap: 块间重叠字符数（保证上下文连续性）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # [#10] 给每个 chunk 分配唯一 ID，用于引用溯源
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def index_document(file_path: str, filename: str = None, agent_id: str = None) -> dict:
    """
    完整的文档索引流程：加载 → 分块 → 分批向量化 → 存储
    分批写入，避免 Embedding API 单次批量超限（智谱限制64条/次）

    Returns:
        dict: 包含分块数量和状态信息
    """
    if filename is None:
        filename = os.path.basename(file_path)

    logger.info(f"开始索引文档: {filename}")

    # 1. 加载文档
    docs = load_document(file_path)

    # 2. 给文档添加元数据
    for doc in docs:
        doc.metadata["source_file"] = filename

    # 3. 分块
    chunks = split_documents(docs)

    if not chunks:
        return {
            "filename": filename,
            "chunks": 0,
            "status": "success",
            "message": f"文档 {filename} 内容为空，无需索引",
        }

    # 4. 分批向量化并存储（每批不超过 EMBEDDING_BATCH_SIZE 条）
    vector_store = get_vector_store(agent_id=agent_id)
    total_chunks = len(chunks)
    batch_count = (total_chunks + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for i in range(batch_count):
        start = i * EMBEDDING_BATCH_SIZE
        end = min(start + EMBEDDING_BATCH_SIZE, total_chunks)
        batch = chunks[start:end]
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            # 如果中途失败，尝试回滚已写入的数据
            try:
                # 查找该文档已写入的分块并删除
                collection = vector_store._collection
                existing = collection.get(
                    where={"source_file": filename},
                    include=["metadatas"],
                )
                if existing.get("ids"):
                    collection.delete(ids=existing["ids"])
            except Exception:
                pass
            raise RuntimeError(f"第 {i+1}/{batch_count} 批向量化失败（分块 {start+1}-{end}）: {str(e)}")

    logger.info(f"文档索引完成: {filename}, 共 {total_chunks} 个分块")    
    return {
        "filename": filename,
        "chunks": total_chunks,
        "status": "success",
        "message": f"文档 {filename} 已成功索引，共 {total_chunks} 个分块（分 {batch_count} 批写入）",
    }


# ===== [#9] 混合检索 + 重排序 =====

def _bm25_keyword_search(query: str, top_k: int = 10, agent_id: str = None) -> list[dict]:
    """
    [#9] BM25 风格的关键词检索（简化版）
    从 ChromaDB 获取全量文档，按关键词匹配度排序
    作为向量检索的补充，提升关键词精确匹配场景的召回率
    """
    vector_store = get_vector_store(agent_id=agent_id)
    try:
        collection = vector_store._collection
        # 获取所有文档
        all_docs = collection.get(include=["documents", "metadatas"])
        
        if not all_docs.get("ids"):
            return []

        query_terms = set(re.findall(r'[\u4e00-\u9fff]+|\w+', query.lower()))
        # 过滤停用词
        stopwords = {'的', '了', '是', '在', '和', '与', '有', '什么', '怎么', '如何', '哪些', '这个', '那个', 'a', 'an', 'the', 'is', 'are', 'was', 'were'}
        query_terms = query_terms - stopwords

        scored = []
        for i, doc_id in enumerate(all_docs["ids"]):
            content = all_docs["documents"][i] or ""
            metadata = all_docs["metadatas"][i] or {}
            
            # 计算关键词匹配分
            content_lower = content.lower()
            match_count = sum(1 for term in query_terms if term in content_lower)
            if match_count == 0:
                continue

            # TF 近似：关键词出现次数 / 文档长度
            tf_score = match_count / max(len(content), 1) * 1000
            
            scored.append({
                "content": content,
                "source": metadata.get("source_file", "未知来源"),
                "chunk_index": metadata.get("chunk_index", -1),
                "bm25_score": tf_score,
                "id": doc_id,
            })

        # 按 BM25 分排序
        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored[:top_k]

    except Exception as e:
        logger.warning(f"BM25 关键词检索失败: {e}")
        return []


def _reciprocal_rank_fusion(vector_results: list[dict], keyword_results: list[dict], k: int = 60) -> list[dict]:
    """
    [#9] 倒数排名融合（Reciprocal Rank Fusion）
    将向量检索和关键词检索的结果融合，按融合分数排序
    
    RRF公式: score = 1/(k + rank_vector) + 1/(k + rank_keyword)
    """
    fused_scores = {}
    
    # 向量检索结果
    for rank, item in enumerate(vector_results):
        content_key = item["content"][:200]  # 用内容前200字作为唯一标识
        if content_key not in fused_scores:
            fused_scores[content_key] = {**item, "rrf_score": 0}
        fused_scores[content_key]["rrf_score"] += 1.0 / (k + rank + 1)
        # 保留向量相似度
        if "relevance_score" not in fused_scores[content_key]:
            fused_scores[content_key]["relevance_score"] = item.get("relevance_score", 0)
    
    # 关键词检索结果
    for rank, item in enumerate(keyword_results):
        content_key = item["content"][:200]
        if content_key not in fused_scores:
            fused_scores[content_key] = {
                "content": item["content"],
                "source": item.get("source", "未知来源"),
                "chunk_index": item.get("chunk_index", -1),
                "relevance_score": 0,
                "rrf_score": 0,
            }
        fused_scores[content_key]["rrf_score"] += 1.0 / (k + rank + 1)
        # 如果有 BM25 分，补充
        if "bm25_score" in item and "bm25_score" not in fused_scores[content_key]:
            fused_scores[content_key]["bm25_score"] = item["bm25_score"]
    
    # 按 RRF 分排序
    results = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return results


def search_documents(query: str, top_k: int = 3, agent_id: str = None) -> list[dict]:
    """
    [#9] 混合检索：向量语义检索 + BM25关键词检索 + RRF融合
    [#10] 引用溯源：返回结果标注文档名 + 段落位置

    Args:
        query: 用户查询
        top_k: 返回最相关的 K 个结果

    Returns:
        list[dict]: 检索结果列表
    """
    # 1. 向量语义检索（按 agent_id 隔离）
    vector_store = get_vector_store(agent_id=agent_id)
    vector_results_raw = []
    try:
        vector_results_raw = vector_store.similarity_search_with_score(query, k=top_k * 3)  # 多取用于融合
    except Exception as e:
        error_str = str(e)
        logger.warning(f"向量检索失败，回退到关键词检索: {e}")
        # 如果是429余额不足错误，记录更详细的提示
        if '429' in error_str or '余额' in error_str or '1113' in error_str:
            logger.warning(f"Embedding API 余额不足（429），向量检索不可用。建议：1）充值智谱API余额 2）或更换embedding模型。当前仅使用关键词检索。")
        vector_results_raw = []

    vector_results = []
    for doc, score in vector_results_raw:
        vector_results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "未知来源"),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "relevance_score": round(1 - score, 4),
        })

    # 2. BM25 关键词检索（按 agent_id 隔离）
    keyword_results = _bm25_keyword_search(query, top_k=top_k * 3, agent_id=agent_id)

    # 3. RRF 融合
    if keyword_results:
        fused_results = _reciprocal_rank_fusion(vector_results, keyword_results)
    else:
        # 关键词检索失败，直接用向量结果
        fused_results = vector_results

    # 4. 格式化输出（取 top_k）
    formatted = []
    for r in fused_results[:top_k]:
        formatted.append({
            "content": r["content"],
            "source": r.get("source", "未知来源"),
            "chunk_index": r.get("chunk_index", -1),
            "relevance_score": r.get("relevance_score", round(r.get("rrf_score", 0), 4)),
        })

    return formatted


def get_document_content(filename: str, agent_id: str = None) -> dict:
    """获取知识库中指定文档的完整内容（从磁盘原始文件读取，不依赖向量搜索）
    
    与 search_documents 不同，此函数返回文档的完整文本内容，
    而不是分块后的片段。用于文档修改前获取完整内容。
    
    Args:
        filename: 文档文件名（含扩展名）
        agent_id: 智能体ID（用于验证文档归属，不参与向量搜索）
    
    Returns:
        dict: 包含文档完整内容、状态信息
    """
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    
    if not os.path.exists(file_path):
        return {
            "filename": filename,
            "status": "not_found",
            "content": "",
            "message": f"文档 {filename} 在服务器上未找到",
        }
    
    try:
        docs = load_document(file_path)
        full_content = "\n".join([doc.page_content for doc in docs])
        
        if not full_content.strip():
            return {
                "filename": filename,
                "status": "empty",
                "content": "",
                "message": f"文档 {filename} 内容为空",
            }
        
        return {
            "filename": filename,
            "status": "success",
            "content": full_content,
            "char_count": len(full_content),
            "message": f"成功获取文档 {filename} 的完整内容（共 {len(full_content)} 字符）",
        }
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "content": "",
            "message": f"读取文档失败: {str(e)}",
        }


def list_indexed_documents(agent_id: str = None) -> list[str]:
    """列出知识库中所有已索引的文档（按 agent_id 隔离）"""
    vector_store = get_vector_store(agent_id=agent_id)
    # 从 ChromaDB 的元数据中提取所有文档名
    try:
        collection = vector_store._collection
        all_docs = collection.get(include=["metadatas"])
        sources = set()
        for meta in all_docs["metadatas"]:
            if meta and "source_file" in meta:
                sources.add(meta["source_file"])
        return sorted(list(sources))
    except Exception:
        return []


def update_document(filename: str, new_content: str, agent_id: str = None, async_reindex: bool = False) -> dict:
    """
    修改知识库中已有文档的内容
    流程：删除旧的向量分块 → 用新内容覆盖原文件 → 重新索引

    Args:
        filename: 要修改的文档文件名
        new_content: 新的文档内容（纯文本）
        agent_id: 智能体ID
        async_reindex: 是否异步重索引（True=立即返回文件已修改，后台执行重索引，大幅加速响应）

    Returns:
        dict: 包含修改状态和详细信息
    """
    # 1. 检查文件是否存在
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if not os.path.exists(file_path):
        return {
            "filename": filename,
            "status": "not_found",
            "message": f"文档 {filename} 在服务器上未找到",
        }

    vector_store = get_vector_store(agent_id=agent_id)
    collection = vector_store._collection

    # 2. 删除旧的向量分块
    chunks_deleted = 0
    try:
        results = collection.get(
            where={"source_file": filename},
            include=["metadatas"],
        )
        chunk_ids = results.get("ids", [])
        if chunk_ids:
            collection.delete(ids=chunk_ids)
            chunks_deleted = len(chunk_ids)
    except Exception as e:
        logger.warning(f"删除旧向量分块时出错: {e}")

    # 3. 用新内容覆盖原文件
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".txt":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        elif ext == ".docx":
            # 对于 docx 文件，用 python-docx 生成新文档
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument()
                # 按换行分段写入
                for line in new_content.split("\n"):
                    doc.add_paragraph(line)
                doc.save(file_path)
            except ImportError:
                # 如果没有 python-docx，回退为纯文本写入
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
        elif ext == ".pdf":
            # PDF 不支持直接修改内容，回退为纯文本写入（改扩展名为 .txt）
            txt_path = file_path.rsplit('.', 1)[0] + '.txt'
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            # 删除原 PDF 文件
            os.remove(file_path)
            filename = filename.rsplit('.', 1)[0] + '.txt'
            file_path = txt_path
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "message": f"写入文件失败: {str(e)}",
        }

    # 4. 重新索引
    if async_reindex:
        # 异步重索引：在后台线程执行，立即返回文件已修改的响应
        import threading
        def _background_reindex(fp, fn, aid):
            try:
                index_result = index_document(fp, fn, agent_id=aid)
                logger.info(f"后台重索引完成: {fn}, {index_result.get('chunks', 0)} 个分块")
            except Exception as e:
                logger.error(f"后台重索引失败: {fn}, {e}")

        thread = threading.Thread(target=_background_reindex, args=(file_path, filename, agent_id), daemon=True)
        thread.start()

        return {
            "filename": filename,
            "status": "success",
            "chunks_deleted": chunks_deleted,
            "chunks_indexed": "后台索引中",
            "message": f"文档 {filename} 已成功修改（删除 {chunks_deleted} 个旧分块，新内容正在后台索引中）",
        }
    else:
        # 同步重索引：等待完成后再返回
        try:
            index_result = index_document(file_path, filename, agent_id=agent_id)
        except Exception as e:
            return {
                "filename": filename,
                "status": "error",
                "message": f"重新索引失败: {str(e)}",
                "chunks_deleted": chunks_deleted,
            }

        return {
            "filename": filename,
            "status": "success",
            "chunks_deleted": chunks_deleted,
            "chunks_indexed": index_result.get("chunks", 0),
            "message": f"文档 {filename} 已成功修改（删除 {chunks_deleted} 个旧分块，重新索引 {index_result.get('chunks', 0)} 个新分块）",
        }


def delete_document(filename: str, agent_id: str = None) -> dict:
    """
    从知识库中删除指定文档
    包括：从 ChromaDB 删除向量分块 + 删除原始文件

    Args:
        filename: 要删除的文档文件名

    Returns:
        dict: 包含删除状态和详细信息
    """
    vector_store = get_vector_store(agent_id=agent_id)
    collection = vector_store._collection

    # 1. 查找该文档的所有分块 ID
    try:
        results = collection.get(
            where={"source_file": filename},
            include=["metadatas"],
        )
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "message": f"查询 ChromaDB 失败: {str(e)}",
        }

    chunk_ids = results.get("ids", [])
    if not chunk_ids:
        return {
            "filename": filename,
            "status": "not_found",
            "message": f"文档 {filename} 在知识库中未找到",
        }

    # 2. 从 ChromaDB 删除所有分块
    try:
        collection.delete(ids=chunk_ids)
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "message": f"从向量数据库删除失败: {str(e)}",
        }

    # 3. 删除原始文件
    file_deleted = False
    file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            file_deleted = True
        except Exception as e:
            return {
                "filename": filename,
                "chunks_deleted": len(chunk_ids),
                "file_deleted": False,
                "status": "partial",
                "message": f"向量分块已删除 {len(chunk_ids)} 个，但原始文件删除失败: {str(e)}",
            }

    return {
        "filename": filename,
        "chunks_deleted": len(chunk_ids),
        "file_deleted": file_deleted,
        "status": "success",
        "message": f"文档 {filename} 已成功删除（{len(chunk_ids)} 个分块，原始文件{'已删除' if file_deleted else '不存在'}）",
    }


def delete_agent_collection(agent_id: str) -> dict:
    """删除智能体的整个知识库 collection
    
    删除智能体时调用，清理 ChromaDB 中该智能体专属的 collection。
    同时清理缓存中的 vector_store 实例。
    
    Args:
        agent_id: 智能体ID
    
    Returns:
        dict: 包含删除状态和详细信息
    """
    if not agent_id:
        return {"status": "error", "message": "agent_id 不能为空"}
    
    import chromadb
    collection_name = _get_collection_name(agent_id)
    
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        # 先检查 collection 是否存在
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name not in existing_collections:
            return {"status": "success", "message": f"智能体知识库 {collection_name} 不存在，无需删除"}
        
        # 删除 collection
        client.delete_collection(collection_name)
        
        # 清理缓存
        cache_key = agent_id or "__global__"
        if cache_key in _vector_store_cache:
            del _vector_store_cache[cache_key]
        
        print(f"[DEBUG-知识库] 已删除智能体 collection: {collection_name}")
        return {"status": "success", "message": f"智能体知识库 {collection_name} 已删除"}
    except Exception as e:
        print(f"[DEBUG-知识库] 删除智能体 collection 失败: {e}")
        return {"status": "error", "message": f"删除知识库失败: {str(e)}"}


def export_document_as_docx(content: str, filename: str, title: str = "") -> dict:
    """
    将文本内容导出为 .docx 文件，保存到下载目录，供用户下载
    
    Args:
        content: 文档内容（纯文本）
        filename: 输出文件名（含扩展名）
        title: 文档标题（可选，将作为文档第一行加粗显示）
    
    Returns:
        dict: 包含导出状态和文件路径
    """
    try:
        from docx import Document as DocxDocument
        from docx.shared import Pt, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        # 回退方案：保存为 .txt
        txt_filename = filename.rsplit('.', 1)[0] + '.txt'
        file_path = os.path.join(settings.DOCUMENTS_DIR, txt_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            if title:
                f.write(f"{title}\n{'=' * len(title.encode('gbk', errors='replace'))}\n\n")
            f.write(content)
        return {
            "status": "success",
            "filename": txt_filename,
            "file_path": file_path,
            "message": f"文档已导出为 {txt_filename}（python-docx 未安装，回退为 txt 格式）",
        }
    
    try:
        file_path = os.path.join(settings.DOCUMENTS_DIR, filename)
        doc = DocxDocument()
        
        # 设置默认字体
        style = doc.styles['Normal']
        font = style.font
        font.name = '宋体'
        font.size = Pt(11)
        
        # 添加标题
        if title:
            heading = doc.add_heading(title, level=1)
            heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            # 从文件名提取标题
            doc_title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ')
            heading = doc.add_heading(doc_title, level=1)
            heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # 按段落写入内容
        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                # 空行保留为空段落
                doc.add_paragraph('')
                continue
            
            # 检测是否为Markdown标题格式
            if line_stripped.startswith('# ') and not line_stripped.startswith('## '):
                # 一级标题（跳过，因为已经有标题了）
                text = line_stripped[2:].strip()
                p = doc.add_heading(text, level=2)
            elif line_stripped.startswith('## '):
                text = line_stripped[3:].strip()
                p = doc.add_heading(text, level=2)
            elif line_stripped.startswith('### '):
                text = line_stripped[4:].strip()
                p = doc.add_heading(text, level=3)
            elif line_stripped.startswith('#### '):
                text = line_stripped[5:].strip()
                p = doc.add_heading(text, level=4)
            elif line_stripped.startswith('- ') or line_stripped.startswith('* '):
                # 列表项
                text = line_stripped[2:].strip()
                # 简单处理Markdown粗体
                text = _clean_markdown_formatting(text)
                doc.add_paragraph(text, style='List Bullet')
            elif line_stripped.startswith('|') and '|' in line_stripped[1:]:
                # 表格行 - 收集连续的表格行
                p = doc.add_paragraph(line_stripped)
                p.style = doc.styles['Normal']
            else:
                # 普通段落
                text = _clean_markdown_formatting(line_stripped)
                doc.add_paragraph(text)
        
        doc.save(file_path)
        
        return {
            "status": "success",
            "filename": filename,
            "file_path": file_path,
            "message": f"文档已导出为 {filename}",
        }
    except Exception as e:
        # 回退为txt
        txt_filename = filename.rsplit('.', 1)[0] + '.txt'
        file_path = os.path.join(settings.DOCUMENTS_DIR, txt_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            if title:
                f.write(f"{title}\n\n")
            f.write(content)
        return {
            "status": "success",
            "filename": txt_filename,
            "file_path": file_path,
            "message": f"文档已导出为 {txt_filename}（docx 生成失败: {str(e)}，回退为 txt 格式）",
        }


def _clean_markdown_formatting(text: str) -> str:
    """清理Markdown格式标记，转为纯文本"""
    import re
    # 去掉粗体标记 **text** → text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # 去掉斜体标记 *text* → text
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # 去掉行内代码 `code` → code
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text


def list_all_collections() -> list[dict]:
    """列出 ChromaDB 中所有的 collection 及其文档数（诊断用）"""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        collections = client.list_collections()
        result = []
        for c in collections:
            try:
                count = c.count()
            except:
                count = -1
            result.append({"name": c.name, "count": count})
        return result
    except Exception as e:
        return [{"name": "error", "count": 0, "message": str(e)}]
