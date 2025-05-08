# LLM Question Assistant 🤖

基于大语言模型的问答助手，结合模型微调与RAG技术，实现智能面试问答与飞书机器人集成

## 主要功能
- 📚 自动化面试数据集构建
- 🧠 领域自适应模型微调
- 🔍 RAG文档增强问答
- 🤖 多轮对话智能体
- ✈️ 飞书机器人无缝对接

## 实现效果图


![f2bfe252-4956-4385-85c3-d2cbfd07334d](https://github.com/user-attachments/assets/45f95af9-600c-4220-8afd-f5cd9844d0d8)

![5b5c3342-f239-44a3-b6bf-a1560ef1e6bd](https://github.com/user-attachments/assets/f7240456-8458-4e1f-ac81-f723e4848d43)



# run
运行main.py就可以，但需要文件结构以及相关的大模型位置。由于使用了lora微调，必须要有lora微调后的模型文件

# 流程
## 一、数据集构建(data文件夹)
## 1：人工提取llm相关问题
收集了B站UP主  文言AI 发布的llm面试相关题，共110条问题，在question.txt文件中。 
文言AI的主页：https://space.bilibili.com/258582830?spm_id_from=333.337.0.0
## 2：通过deepseek调用回答
将问题发给deepseek，调用deepseek api进行回答，将问题和回答保存为question， answer对的json文件。(ques_ans_ge.py)
回答的prompt:"问题的回答大约 400 - 600 字左右，当文字已足够回答内容时字数不足也可以，当问题较宽泛，需要举例等方式进行回答时，回答字数可以超，但不要超过 800 字。问题：{}"
此部分需要deepseek api或依赖其他模型生成。
## 3：清理answer中的垃圾字符。
使用data_preprocess.py文件，最后的结果为train.jsonl文件，这将接下来用于few-shot deepseek1.5b


## 二、模型微调 (model文件夹)
通过train.py使用生成的数据集使用lora few-shot微调 deepseek r1 1.5b。
## 三、搭建rag系统(rag文件夹)
同时将train.jsonl 先读为
问题：{}
答案：{}
然后使用faiss构建向量数据库
## 四、部署 (deploy文件夹)
## 4.1 agent
使用模型和rag系统进行agent构建
## 4.2 feishu
agent接入飞书机器人 (feishu.py需要调用飞书应用机器人的api)
## 4.3 config
 所有部署阶段的超参数以及api等等在此处一同修改

