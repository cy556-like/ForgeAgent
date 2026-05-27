---
Task ID: 1
Agent: Main Agent
Task: 诊断429错误并修复modify_document_tool无法获取完整文档内容的问题

Work Log:
- 分析429错误原因：embedding-3模型与聊天模型(glm-4.7)共用API Key但额度独立，embedding额度已耗尽
- 在document.py中添加get_document_content()函数，直接从磁盘读取完整文档内容（不依赖向量搜索，不消耗embedding额度）
- 在tools.py中添加get_document_content_tool工具，让模型能获取知识库中指定文档的完整全文
- 更新modify_document_tool描述，添加操作流程说明：先调用get_document_content_tool获取完整内容，修改后再调用modify_document_tool保存
- 将get_document_content_tool加入BASE_TOOLS列表
- 在core.py中添加"get_document_content_tool": "获取文档全文"的中文映射
- 优化search_documents_tool对429错误的处理，当embedding余额不足时给出明确提示并引导使用get_document_content_tool

Stage Summary:
- 新增get_document_content()函数（document.py），从磁盘读取完整文档，不依赖向量搜索
- 新增get_document_content_tool工具（tools.py），模型可直接获取文档完整内容
- 修改文档的正确流程：get_document_content_tool获取全文 → 在全文基础上修改 → modify_document_tool保存
- 429错误处理优化：明确提示embedding余额不足，引导使用不消耗embedding额度的get_document_content_tool
- 部署文件已复制到download目录：tools.py, document.py, core.py
