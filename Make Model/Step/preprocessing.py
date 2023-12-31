# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:05:26 2023

@author: admin
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib

class Preprocess():
    
    def __init__(self, csv, model):
        self.csv = csv
        self.model = model

    # 라벨 인코딩, 원-핫 인코딩
    def encoding(self, columnN):
    
        self.le = LabelEncoder()
        self.le.fit(self.csv[columnN])
        self.csv[columnN] = self.le.transform(self.csv[columnN])#라벨인코딩
    
        encoder = OneHotEncoder()
        self.Y = encoder.fit_transform(self.csv[[columnN]]).toarray()
        
    
    # 토큰화 및 문장 표현 생성
    def tokenizer(self):
        
        master_dict = self.csv['Question'].tolist()
        self.X = self.model.encode(master_dict, convert_to_tensor=True) # 마스터 사전에 있는 모든 질문의 표현을 계산        
        
    # 학습 전용 데이터와 테스트 전용 데이터로 나누기
    def split_data(self):

        # Split data into train and test sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
    
    # PyTorch의 torch.float32 데이터 타입을 TensorFlow에서 사용하는 DType으로 변환
    def change_data_type(self):

        X_train_array = self.X_train.cpu().detach().numpy() # PyTorch Tensor를 NumPy 배열로 변환
        X_test_array = self.X_test.cpu().detach().numpy() # PyTorch Tensor를 NumPy 배열로 변환

        X_train = tf.convert_to_tensor(X_train_array, dtype=tf.float32) # NumPy 배열을 TensorFlow Tensor로 변환
        X_test = tf.convert_to_tensor(X_test_array, dtype=tf.float32) # NumPy 배열을 TensorFlow Tensor로 변환

        self.X_train = tf.identity(X_train) #gpu로 이동
        self.X_test = tf.identity(X_test) #gpu로 이동
        
    def embedding_size(self):
        _, embedding_size = self.X.shape
        return embedding_size
    
    def categories(self):
        
        n_categories = self.Y.shape[1]
        return n_categories
    
    def label_encoder(self):
        
        return self.le
    
    def train_data(self):
        
        return self.X_train, self.Y_train
    
    def test_data(self):
        
        return self.X_test, self.Y_test
    
    def data_step_ing(self, name):
        self.encoding(name) # 라벨 인코딩, 원-핫 인코딩
        self.tokenizer() # 토큰화 및 문장 표현 생성
        self.split_data() # 데이터 split
        self.change_data_type() # 데이터 형식 변경