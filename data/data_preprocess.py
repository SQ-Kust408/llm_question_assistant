import json
import re

def clean_answer(answer):
    """深度清洗回答文本，去除所有冗余符号并紧凑化"""
    # 1. 去除所有Markdown格式符号
    answer = re.sub(r'\*\*|###|---|\\|\$', '', answer)  # 去除 **、###、---、\\、$
    answer = re.sub(r'#+', '', answer)  # 去除单个或多个#（处理###后的残留）

    # 2. 处理空白字符（换行、制表符、连续空格）
    answer = re.sub(r'\\n|\\r|\\t|\n|\r|\t', ' ', answer)  # 转义字符转空格
    answer = re.sub(r' +', ' ', answer).strip()  # 合并连续空格并去首尾空格

    # 3. 处理特殊符号（保留必要标点，仅去除格式相关符号）
    # 可根据需求添加更多需要去除的符号（如保留-作为连接符，去除连续的---）
    return answer

# 读取原始数据（假设data.json在当前目录）
with open('data.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# 清洗每个样本并转换为LoRA训练所需格式
lora_dataset = []
for sample in dataset:
    cleaned_answer = clean_answer(sample['answer'])
    prompt = f"问题：{sample['question']}\n答案："
    completion = cleaned_answer
    lora_dataset.append({
        "prompt": prompt,
        "completion": completion
    })

# 保存为JSONL格式（每行一个样本，适合大规模训练）
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in lora_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"成功生成 LoRA 训练数据：train.jsonl，包含 {len(lora_dataset)} 条样本")