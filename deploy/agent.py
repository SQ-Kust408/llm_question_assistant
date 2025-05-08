# agent.py
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from Interview_assistant.rag.rag import load_rag_database
from config import RAG_CONFIG, MODEL_CONFIG
from peft import PeftModel
import os

class Agent:
    def __init__(self):
        # 加载基础模型
        base_model_path = MODEL_CONFIG["BASE_MODEL"]  # 正确读取配置中的路径
        print(f"基础模型路径：{os.path.abspath(base_model_path)}")
        print(f"适配器路径：{os.path.abspath(MODEL_CONFIG['MODEL_PATH'])}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=MODEL_CONFIG["DEVICE_MAP"],
            torch_dtype=getattr(torch, MODEL_CONFIG["TORCH_DTYPE"])
        )

        # 加载适配器
        self.model = PeftModel.from_pretrained(
            base_model,
            MODEL_CONFIG["MODEL_PATH"],  #
            adapter_name="lora_adapter"
        )
        self.model = self.model.merge_and_unload()  # 合并适配器（可选）

        # 初始化检索系统
        self.rag_index, self.rag_docs = load_rag_database(
            RAG_CONFIG["INDEX_PATH"],
            RAG_CONFIG["DOCS_PATH"]
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def _retrieve_context(self, question: str) -> str:
        # """统一使用RAG检索"""
        # query_embedding = self.encoder.encode([question])
        # distances, indices = self.rag_index.search(query_embedding, RAG_CONFIG["TOP_K"])
        # return "\n".join([self.rag_docs[i] for i in indices[0] if i < len(self.rag_docs)])
        """改进后的混合检索"""
        # RAG检索
        query_embedding = self.encoder.encode([question])
        distances, indices = self.rag_index.search(query_embedding, RAG_CONFIG["TOP_K"])
        rag_ctx = "\n".join([self.rag_docs[i] for i in indices[0] if i < len(self.rag_docs)])
        # MCP检索（当检测到Java问题时）
        mcp_ctx = ""

        return f"知识库上下文：\n{rag_ctx}\n\n官方文档参考：\n{mcp_ctx}"

    def get_answer(self, question: str) -> str:
        try:
            context = self._retrieve_context(question)
            prompt = f"""基于以下上下文给出专业解答：
            {context}

            问题：{question}
            答案："""

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1500,
                truncation=True
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MODEL_CONFIG["MAX_NEW_TOKENS"],
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )

            return self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).split("答案：")[-1].strip()

        except Exception as e:
            print(f"生成错误: {str(e)}")
            return "系统暂时无法回答这个问题，请尝试重新提问。"


if __name__ == "__main__":
    agent = Agent()