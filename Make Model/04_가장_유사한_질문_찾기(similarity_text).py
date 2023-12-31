# -*- coding: utf-8 -*-
"""04. 가장 유사한 질문 찾기(Similarity text).ipynb

- distiluse-base-multilingual-cased : 아랍어, 중국어, 네덜란드어, 영어, 프랑스어, 독일어, 이탈리아어, 한국어, 폴란드어, 포르투갈어, 러시아어, 스페인어 및 터키어를 지원
- xlm-r-base-en-ko-nli-ststb : 한국어와 영어를 지원
- xlm-r-large-en-ko-nli-ststb : 한국어와 영어를 지원
"""

# 필요한 모듈 import
from Step.load_data import load_model

# 모델, DB loading
model, csv = load_model()

# 모든 질문 DB의 문장 표현 계산

master_dict_repre = {}
master_dict = {}
intent_type = ['정보', '외모', '성격', '관리', '먹이', '의료']
species_type = ['강아지', '고양이', '앵무새', '햄스터', '기타']

original_data = csv.copy()

for s_type in species_type:
    
    master_dict_temp = {}
    master_dict_repre_temp = {}
    for i_type in intent_type:
        csv = csv[csv['Species']==s_type]
        type_dict = csv[csv['Intent']==i_type]['Question'].tolist()
        
        if not type_dict:
            master_dict_temp[i_type]=[] # 질문 저장
            master_dict_repre_temp[i_type]=[] # 표현 저장
        
        else:
    
            master_dict_temp[i_type]=type_dict # 질문 저장
            type_dict_repre = model.encode(type_dict, convert_to_tensor=True) # 마스터 사전에 있는 모든 질문의 표현을 계산
            master_dict_repre_temp[i_type]=type_dict_repre # 표현 저장
        
        csv = original_data
    
    master_dict_repre[s_type]=master_dict_repre_temp # 표현 저장
    master_dict[s_type]=master_dict_temp # 표현 저장
    
import joblib

route='Model/{}'
joblib.dump(master_dict_repre, filename =route.format('master_dict_repre.pkl'))   #pickle 파일
joblib.dump(master_dict, filename =route.format('master_dict.pkl'))   #pickle 파일
