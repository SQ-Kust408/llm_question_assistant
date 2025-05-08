# config.py
import os

# 获取项目根目录（Interview_assistant）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型服务配置
MODEL_CONFIG = {
    "BASE_MODEL": os.path.join(BASE_DIR, "model", "deepseek-r1-1.5b"),
    "MODEL_PATH": os.path.join(BASE_DIR, "model", "checkpoint-130"),
    "DEVICE_MAP": "auto",
    "TORCH_DTYPE": "bfloat16",
    "MAX_NEW_TOKENS": 1024
}
# 在 MODEL_CONFIG 后添加验证代码
import os

assert os.path.exists(MODEL_CONFIG["BASE_MODEL"]), f"基础模型路径不存在：{MODEL_CONFIG['BASE_MODEL']}"
assert os.path.exists(MODEL_CONFIG["MODEL_PATH"]), f"适配器路径不存在：{MODEL_CONFIG['MODEL_PATH']}"

# 检索增强配置
RAG_CONFIG = {
    "INDEX_PATH": os.path.join(BASE_DIR, "rag", "db", "rag_index.faiss"),
    "DOCS_PATH": os.path.join(BASE_DIR, "rag", "db", "rag_documents.pkl"),
    "TOP_K": 3,
    "SCORE_THRESHOLD": 0.85
}

# 飞书接口配置
FEISHU_CONFIG = {
    "APP_ID": "",
    "APP_SECRET": "",
    "RECEIVE_ID": "",
    "RECEIVE_TYPE": "chat_id",
    "MESSAGE_FORMAT": {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": "知识助手",
                "template": "blue"
            }
        }
    }
}

# 系统运行时配置
RUNTIME_CONFIG = {
    "LOG_LEVEL": "INFO",  # 日志级别
    "CACHE_ENABLED": True,  # 是否启用缓存
    "SAFETY_FILTER": True  # 是否启用内容安全过滤
}