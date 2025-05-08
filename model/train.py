import json
import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# 加载分词器（添加padding_side设置）
tokenizer = AutoTokenizer.from_pretrained("deepseek-r1-1.5b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 修改后的数据转换函数
def convert_to_model_inputs(item):
    full_text = item["prompt"] + item["completion"]
    inputs = tokenizer(
        full_text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # 创建标签掩码
    prompt = tokenizer(
        item["prompt"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = inputs["input_ids"].clone()
    labels[0, :prompt["input_ids"].shape[1]] = -100

    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": labels.squeeze()
    }

# 读取JSONL文件
lora_dataset = []
with open('train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        lora_dataset.append(json.loads(line))

# 将 lora_dataset 转换为模型可接受的格式
train_dataset = [convert_to_model_inputs(item) for item in lora_dataset]

# 使用数据收集器处理填充
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 修改后的训练参数
training_args = TrainingArguments(
    output_dir="./tuned_model",
    num_train_epochs=7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    optim="adamw_torch",
    logging_steps=1,  # 每个步骤记录一次日志
    save_strategy="epoch",  # 每个 epoch 保存一次
    save_total_limit=1,     # 只保留最新的一个检查点
    bf16=True,  # 使用bfloat16代替fp16
    report_to="none"
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-r1-1.5b",
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 应用 LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# 开始训练
train_result = trainer.train()

# 提取训练损失
train_losses = []
for log in trainer.state.log_history:
    if 'loss' in log:
        train_losses.append(log['loss'])

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_curve.png')
plt.show()