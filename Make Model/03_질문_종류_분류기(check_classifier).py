# -*- coding: utf-8 -*-
"""03. 인사/애완동물 분류기(Intent classifier)-v2.ipynb

# ■ KoBERT를 활용해 의도 분류기 만들기

#### 필요한 모듈 import
"""

from Step.load_data import load_model
from Step.preprocessing import Preprocess
from Step.model import Model
import pandas as pd

# 데이터 가져오기
def get_data():

  route = 'Data/ChatbotData.csv'
  csv = pd.read_csv(route, encoding = 'utf-8')

  return csv

# 데이터 전처리
def preprocess_dataset(csv, command_csv):
    
    csv['Class'] = '동물'
    command_csv['Class'] = '인사'

    csv = csv.drop(columns=['Intent', 'Species', 'Answer'])

    command_csv = command_csv.drop(columns=['A', 'label'])
    command_csv.columns = ['Question', 'Class']

    csv = pd.concat([csv, command_csv], ignore_index=True)
    
    return csv

# 모델, DB loading
model, csv = load_model()
command_csv = get_data()


# 데이터 전처리
name = 'Class'
csv = preprocess_dataset(csv, command_csv)
pre = Preprocess(csv, model) # 객체 생성
pre.data_step_ing(name)

# 모델 생성
n_categories = pre.categories()
embedding_size = pre.embedding_size()
model = Model(n_categories, embedding_size)

Dense_size = [400, 256, 128, 64, 32]
Dropout_size = [0.2, 0.1, 0.1, 0.2, 0.1]
model.make_model(Dense_size, Dropout_size)

# 모델 학습
epochs = 60 # Set training parameters
batch_size = 512 
train_data = pre.train_data() # 학습 데이터 가져오기

model.model_train(train_data, epochs, batch_size)

# 모델 평가
test_data = pre.test_data()
model.model_evaluate(test_data)

# 모델 저장
route='Model/command.keras'
model.model_save(route)

#파일에 학습 결과 저장
import joblib

route='Model/command_label.pkl'
le = pre.label_encoder()
joblib.dump(le, filename =route)   #pickle 파일
