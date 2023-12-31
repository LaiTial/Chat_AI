# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:02:32 2023

@author: admin
"""

from sentence_transformers import SentenceTransformer, util
from Step.connectDB import HandleDB
import pandas as pd

def load_model():

    # 토크나이저, 모델 다운
    route='../Model/sentence_bert'
    model = SentenceTransformer(route) # 모델 로드
    
    # 데이터 불러오기
    db = HandleDB()
    csv = pd.DataFrame(db.get_data(), columns=['Id', 'Intent', 'Species', 'Question', 'Answer'])
    csv = csv.set_index('Id')
    
    return model, csv