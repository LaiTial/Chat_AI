# 02. 의도 분류기(Intent classifier)-v2.ipynb

# ■ KoBERT를 활용해 의도 분류기 만들기

#### 필요한 모듈 import
from Step.load_data import load_model
from Step.preprocessing import Preprocess
from Step.model import Model
import joblib

# 모델, DB loading
model, csv = load_model()

# 데이터 전처리
name = 'Species'
pre = Preprocess(csv, model) # 객체 생성
pre.data_step_ing(name)

# 모델 생성
n_categories = pre.categories()
embedding_size = pre.embedding_size()
model = Model(n_categories, embedding_size)

Dense_size = [800, 512, 256, 128, 64, 32, 16]
Dropout_size = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]
model.make_model(Dense_size, Dropout_size)

# 모델 학습
epochs = 100 # Set training parameters
batch_size = 256 
train_data = pre.train_data() # 학습 데이터 가져오기

model.model_train(train_data, epochs, batch_size)

# 모델 평가
test_data = pre.test_data()
model.model_evaluate(test_data)

# 모델 저장
route='Model/species.keras'
model.model_save(route)

#파일에 학습 결과 저장

route='Model/species_label.pkl'
le = pre.label_encoder()
joblib.dump(le, filename =route)   #pickle 파일
