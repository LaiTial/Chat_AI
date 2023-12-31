"""04. 가장 유사한 질문 찾기(Similarity text).ipynb

- distiluse-base-multilingual-cased : 아랍어, 중국어, 네덜란드어, 영어, 프랑스어, 독일어, 이탈리아어, 한국어, 폴란드어, 포르투갈어, 러시아어, 스페인어 및 터키어를 지원
- xlm-r-base-en-ko-nli-ststb : 한국어와 영어를 지원
- xlm-r-large-en-ko-nli-ststb : 한국어와 영어를 지원
"""

# 필요한 모듈 import
from sentence_transformers import util

# 문장 표현 계산

def predict_similarity(inp_Q, modelZip):

    model, master_dict, master_dict_repre = modelZip
    
    Q_repre = model.encode(inp_Q, convert_to_tensor=True) # 입력 질문의 표현을 계산
    
    # 입력 질문 표현과 마스터 사전에 있는 모든 질문 표현의 코사인 유사도를 계산
    similarity = util.pytorch_cos_sim(Q_repre,master_dict_repre)
    
    # 가장 유사한 질문을 출력
    import numpy as np
    
    Q = master_dict[np.argmax(similarity)]
    
    # 가장 유사한 질문의 문장 표현 유사도 계산  
    sentence1_representation = Q_repre # 입력 질문 표현
    sentence2_representation = model.encode(Q) # encode 함수로 문장 표현 계산
    
    # 코사인 유사도 구하기
    
    cosine_sim = util.pytorch_cos_sim(sentence1_representation, sentence2_representation)
    
    return Q, cosine_sim # 두 문장의 유사도
