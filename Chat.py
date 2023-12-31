# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:02:47 2023

@author: asna9
"""

import torch
from Predict.intent_predict import predict_class
from Predict.similarity_predict import predict_similarity
from Predict.species_predict import predict_species
from Predict.load_zip import load_data
from Predict.connectDB import HandleDB
from Predict.generate_daily import generate_daily_A
from Predict.generate_animail import generate_animal_A

# 데이터 로딩
representation_model, command_le, command_model, species_le, species_class_model, le, class_model, master_dict, master_dict_repre = load_data() # 데이터 로딩

def chatBot(sentence):

    # 01. 데이터 로딩
    db = HandleDB() # DB 객체 생성
    
    # 01. 질문 class 분류
    modelZip = (representation_model, command_le, command_model) # 전달할 분류 모델 데이터 하나로 묶기
    check_labels = predict_species(sentence, modelZip)
    
    if(check_labels[0]=='인사'):
        return generate_daily_A(sentence)
    
    # 03. 질문 종 분류
    modelZip = (representation_model, species_le, species_class_model) # 전달할 분류 모델 데이터 하나로 묶기
    species_labels = predict_species(sentence, modelZip)
    
    # 04. 질문 의도 분류
    modelZip = (representation_model, le, class_model) # 전달할 분류 모델 데이터 하나로 묶기
    labels = predict_class(sentence, modelZip) # 분류 모델 예측 결과 얻기
    
    # 05. 가장 가능성 높은 2개의 분류에 따른 데이터 결합
    dict_repre = master_dict_repre[species_labels[0]]
    tensor1 = dict_repre[labels[0]] 
    tensor2 = dict_repre[labels[1]]
    
    dict_repre = torch.cat((tensor1, tensor2), dim=0)
    
    dicts = master_dict[species_labels[0]]
    dicts = dicts[labels[0]] + dicts[labels[1]]
    
    # 가장 유사한 질문 얻기
    modelZip = (representation_model, dicts, dict_repre)
    Q, score = predict_similarity(sentence, modelZip)
    A = db.find_answer(Q) # 정답 얻기
    
    print("Type:", check_labels[0])
    print("NER:", species_labels[0])
    print("Intent", labels)
    print("Q:", Q)
    print("A:", A)
    print("Score:", score[0][0])
    
    if(score[0][0] < 0.75):
        generate_animal_A(sentence, labels[0], species_labels[0])
    
    return A
