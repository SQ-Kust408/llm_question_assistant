# rag.py
import os
import json
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def read_documents(file_path: str) -> list:
    """读取并解析train.jsonl文件"""
    documents = []

    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="正在解析JSONL"):
                try:
                    data = json.loads(line.strip())
                    # 组合prompt和completion作为文档内容
                    doc = f"问题：{data['prompt']}\n答案：{data['completion']}"
                    documents.append(doc)
                except json.JSONDecodeError:
                    print(f"警告：跳过无法解析的行：{line[:50]}...")
        return documents
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return []


def vectorize_documents(documents: list) -> np.ndarray:
    """生成文档向量"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("正在生成文档嵌入向量...")
    return model.encode(documents, show_progress_bar=True)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """构建FAISS索引"""
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("嵌入向量必须是二维numpy数组")

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)

    # 转换为float32类型
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    index.add(embeddings)
    return index


def save_rag_database(
        index: faiss.Index,
        documents: list,
        index_path: str = 'db/rag_index.faiss',
        docs_path: str = 'db/rag_documents.pkl'
):
    """保存RAG数据库"""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    print(f"正在保存索引到 {index_path}...")
    faiss.write_index(index, index_path)

    print(f"正在保存文档到 {docs_path}...")
    with open(docs_path, 'wb') as f:
        pickle.dump(documents, f)


def load_rag_database(
        index_path: str = 'db/rag_index.faiss',
        docs_path: str = 'db/rag_documents.pkl'
) -> tuple:
    """加载RAG数据库"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件 {index_path} 不存在")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"文档文件 {docs_path} 不存在")

    print("正在加载索引...")
    index = faiss.read_index(index_path)

    print("正在加载文档...")
    with open(docs_path, 'rb') as f:
        documents = pickle.load(f)

    return index, documents


def llm_retrieve(
        query: str,
        index: faiss.Index,
        documents: list,
        k: int = 5
) -> list:
    """检索最相关的文档"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    # FAISS需要float32输入
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)

    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0] if i < len(documents)]


def main():
    # 配置文件路径
    jsonl_path = 'train.jsonl'
    index_path = 'db/rag_index.faiss'
    docs_path = 'db/rag_documents.pkl'

    # 构建流程
    print("=== 开始构建RAG数据库 ===")

    # 步骤1：读取文档
    documents = read_documents(jsonl_path)
    if not documents:
        return

    print(f"成功读取 {len(documents)} 个文档")

    # 步骤2：向量化
    embeddings = vectorize_documents(documents)

    # 步骤3：构建索引
    try:
        index = build_index(embeddings)
    except ValueError as e:
        print(f"索引构建失败：{str(e)}")
        return

    # 步骤4：保存数据库
    save_rag_database(index, documents, index_path, docs_path)

    # 验证流程
    print("\n=== 验证检索功能 ===")
    test_query = "Java中的多线程如何实现？"

    try:
        loaded_index, loaded_docs = load_rag_database(index_path, docs_path)
        results = llm_retrieve(test_query, loaded_index, loaded_docs)

        print(f"针对查询『{test_query}』的检索结果：")
        for i, doc in enumerate(results, 1):
            print(f"\n结果 {i}:\n{doc[:200]}...")

    except Exception as e:
        print(f"验证失败：{str(e)}")


if __name__ == "__main__":
    main()
