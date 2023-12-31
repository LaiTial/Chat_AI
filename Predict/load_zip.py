# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:14:25 2023

@author: asna9
"""

from sentence_transformers import SentenceTransformer
import joblib
from tensorflow.keras.models import load_model

def load_data():
    
    # 종 분류 모델 불러오기
    
    route = 'Model/{}'
    
    representation_model = SentenceTransformer(route.format('sentence_bert')) # 표현 모델 load
    species_le= joblib.load(route.format('species/species_label.pkl')) # 라벨 인코딩 객체 load   
    species_class_model=load_model(route.format('species/species.keras')) # 분류기 모델 load 
    
    # 질문 의도 분류 모델 불러오기
    command_le= joblib.load(route.format('class/command_label.pkl')) # 라벨 인코딩 객체 load   
    command_model=load_model(route.format('class/command.keras')) # 분류기 모델 load 
    
    # 의도 분류기 모델 불러오기

    le= joblib.load(route.format('intent/intent_v2.pkl')) # 라벨 인코딩 객체 load   
    class_model=load_model(route.format('intent/intent_v2.keras')) # 분류기 모델 load 
    
    # 유사도 계산 모델 load
    master_dict= joblib.load(route.format('master_dict.pkl')) # 질문 list   
    master_dict_repre= joblib.load(route.format('master_dict_repre.pkl')) # 질문 문장표현들 
    
    return representation_model, command_le, command_model, species_le, species_class_model, le, class_model, master_dict, master_dict_repre
