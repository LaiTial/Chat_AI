# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:38:16 2023

@author: asna9
"""

from flask import Flask, request  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
from flask_cors import CORS # 다른 도메인에서 자원을 요청할 때 보안 정책을 적용
import Chat


app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록
CORS(app)

@api.route('/chat')  # 데코레이터 이용, '/chat' 경로에 함수 등록
class ChatResource(Resource):
    def get(self):
        Q = request.args.get('Q')

        answer = Chat.chatBot(Q)
        return {"chat": answer}

api.add_resource(ChatResource, '/chat')

if __name__ == "__main__":
    app.run(host='0.0.0.0') #debug=True, host='0.0.0.0', port=500