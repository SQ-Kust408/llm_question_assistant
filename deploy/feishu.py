# feishu.py
import json
import lark_oapi as lark
from config import FEISHU_CONFIG


def send_message_to_feishu(question: str, answer: str) -> bool:
    """
    发送问答结果到飞书
    返回状态：True发送成功 / False发送失败
    """
    try:
        client = lark.Client.builder() \
            .app_id(FEISHU_CONFIG["APP_ID"]) \
            .app_secret(FEISHU_CONFIG["APP_SECRET"]) \
            .build()

        content = json.dumps({
            "text": f"❓用户提问：{question}\n\n💡系统回答：{answer[:500]}"  # 限制消息长度
        })

        req = lark.api.im.v1.CreateMessageRequest.builder() \
            .receive_id_type(FEISHU_CONFIG["RECEIVE_TYPE"]) \
            .request_body(lark.api.im.v1.CreateMessageRequestBody.builder()
                          .receive_id(FEISHU_CONFIG["RECEIVE_ID"])
                          .msg_type("text")
                          .content(content)
                          .build()) \
            .build()

        resp = client.im.v1.message.create(req)
        return resp.success()

    except Exception as e:
        print(f"飞书消息发送异常：{str(e)}")
        return False
