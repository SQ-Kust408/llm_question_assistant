# Interview Assistant 🤖

基于大语言模型的面试助手，结合模型微调与RAG技术，实现智能面试问答与飞书机器人集成

## 主要功能
- 📚 自动化面试数据集构建
- 🧠 领域自适应模型微调
- 🔍 RAG文档增强问答
- 🤖 多轮对话智能体
- ✈️ 飞书机器人无缝对接

## 安装依赖
```bash
DEEPSEEK_API_KEY=your_api_key_here
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret

pip install -r requirements.txt
```

## 数据集构建
python ques_ans_gen.py  # 生成基础问答对
python data_process.py  # 处理数据格式匹配，生成数据集(data.json)
## 模型微调 (train.py)
python train.py --data_path data.json --epochs 3
## 搭建rag系统(rag.py)
python rag.py --docs_path ./knowledge_base
## agent构建 (agent.py)
1：使用模型和rag系统进行agent构建
2：保存agent
## 调用agent
python agent.py --model_path ./fine_tuned_model
## agent接入飞书机器人 (feishu.py需要调用飞书应用机器人的api)
python feishu.py --port 8000
Interview_assistant/
├── configs/              # 配置文件
├── knowledge_base/       # RAG文档库
├── trained_models/       # 微调后的模型
├── ques_ans_gen.py       # 问答生成脚本
├── train.py              # 训练脚本
├── rag.py                # RAG系统
├── agent.py              # 对话智能体
└── feishu.py             # 飞书机器人接口