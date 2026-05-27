"""
ForgeAgent API Key 诊断脚本
在阿里云ESC上运行: python test_api_key.py
"""
import os
from openai import OpenAI

API_KEY = "293f63eb22024c179357d83a3e5bc8df.3ugVOlGrYIMbcsxZ"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# 要测试的模型列表
MODELS = [
    "glm-4-flash",     # 免费
    "glm-4-air",       # 便宜
    "glm-4-plus",      # 较贵
    "glm-5.1",         # 最贵
]

print("=" * 60)
print("ForgeAgent API Key 诊断工具")
print("=" * 60)
print(f"\nAPI Key: {API_KEY[:8]}...{API_KEY[-6:]}")
print(f"Base URL: {BASE_URL}")
print()

for model in MODELS:
    print(f"\n--- 测试模型: {model} ---")
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "你好，请回复'测试成功'"}],
            max_tokens=50,
        )
        reply = response.choices[0].message.content
        print(f"  ✅ 成功! 回复: {reply}")
    except Exception as e:
        error_msg = str(e)
        if "1113" in error_msg:
            print(f"  ❌ 余额不足 (code 1113) - 该模型无可用资源包")
        elif "429" in error_msg:
            print(f"  ❌ 限流 (429) - 请求过于频繁")
        elif "401" in error_msg:
            print(f"  ❌ 认证失败 (401) - API Key 无效")
        else:
            print(f"  ❌ 失败: {error_msg[:100]}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
print("\n如果所有模型都失败，说明：")
print("1. API Key 可能已过期或被禁用")
print("2. 账户余额确实为0（请到 https://open.bigmodel.cn 检查）")
print("3. 如果只有部分模型失败，请在 .env 中指定可用的模型")
