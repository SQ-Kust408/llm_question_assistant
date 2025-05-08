import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("deepseek-r1-1.5b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 加载微调后的模型
model = AutoModelForCausalLM.from_pretrained(
    "checkpoint-130",
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 设置模型为评估模式
model.eval()

# 输入提示
prompt = "你的输入提示内容"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# 生成文本
with torch.no_grad():
    output = model.generate(input_ids, max_length=512, num_beams=5, no_repeat_ngram_size=2)

# 解码输出
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)