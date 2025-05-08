from flask import Flask, request, render_template, jsonify, send_from_directory
from agent import Agent
from feishu import send_message_to_feishu
from config import FEISHU_CONFIG, RUNTIME_CONFIG
import threading

app = Flask(__name__)
agent = Agent()

@app.route('/health')
def health_check():
    return jsonify({
        "status": "ok"
    })

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

def async_send_feishu(question: str, answer: str):
    """异步发送飞书消息"""
    if FEISHU_CONFIG["APP_ID"] and FEISHU_CONFIG["APP_SECRET"]:
        success = send_message_to_feishu(question, answer)
        if RUNTIME_CONFIG["LOG_LEVEL"] == "DEBUG":
            print(f"飞书消息发送状态：{'成功' if success else '失败'}")

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({"error": "问题内容不能为空"}), 400
        try:
            answer = agent.get_answer(question)
            # 异步发送飞书通知
            threading.Thread(target=async_send_feishu, args=(question, answer)).start()
            return jsonify({
                "question": question,
                "answer": answer
            })
        except Exception as e:
            return jsonify({
                "error": f"处理请求时发生错误：{str(e)}"
            }), 500
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=RUNTIME_CONFIG["LOG_LEVEL"] == "DEBUG"
    )