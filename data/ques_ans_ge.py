import json
import requests

DEEPSEEK_API_KEY = ""
QUESTION_FILE = "question.txt"
OUTPUT_FILE = "data.json"
PROMPT_TEMPLATE = "问题的回答大约 400 - 600 字左右，当文字已足够回答内容时字数不足也可以，当问题较宽泛，需要举例等方式进行回答时，回答字数可以超，但不要超过 800 字。问题：{}"
API_URL = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}


def generate_answer(question: str) -> str | None:
    """根据问题生成答案，包含定制化Prompt"""
    prompt = PROMPT_TEMPLATE.format(question.strip())
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        print(f"[HTTP错误] {question}: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"[处理异常] {question}: {str(e)[:50]}")  # 缩短错误信息便于查看
        return None


def load_questions() -> list[str]:
    """从文件加载问题列表，过滤无效行"""
    try:
        with open(QUESTION_FILE, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]  # 过滤空行和空白行
        if not questions:
            print(f"[警告] {QUESTION_FILE} 中无有效问题")
        return questions
    except FileNotFoundError:
        print(f"[错误] 未找到问题文件: {QUESTION_FILE}")
        return []
    except Exception as e:
        print(f"[文件读取错误] {str(e)[:50]}")
        return []


def build_dataset(questions: list[str]) -> None:
    """构建问题-答案数据集并保存为JSON"""
    dataset = []
    for idx, question in enumerate(questions, 1):
        print(f"[处理中 {idx}/{len(questions)}] {question[:30]}...")  # 显示问题前30字
        answer = generate_answer(question)
        if answer:
            dataset.append({
                "question": question,
                "answer": answer
            })
    # 保存JSON文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"\n[完成] 数据已保存到 {OUTPUT_FILE}")
    print(f"有效条目: {len(dataset)} / 总问题数: {len(questions)}")


if __name__ == "__main__":
    # 1. 加载问题列表
    questions = load_questions()
    if not questions:
        print("[错误] 未获取到有效问题，程序终止")
        exit(1)

    # 2. 批量处理问题
    print(f"开始处理 {len(questions)} 个问题...\n")
    build_dataset(questions)