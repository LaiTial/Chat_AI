# 01. 종 분류 예측
# 필요한 모듈 import
import tensorflow as tf

def predict_species(sentence, modelZip):
    
    # 토크나이저, 모델, 라벨 인코딩 객체 불러오기
    representation_model, le, model = modelZip

    # 임베딩
    predict_X = representation_model.encode(sentence, convert_to_tensor=True) # 입력 질문의 표현을 계산
    
    # shape로 이차원으로 변경
    predict_X = predict_X.view(1, -1)
    
    # 데이터 타입 변환
    
    arr = predict_X.detach().numpy() # PyTorch Tensor를 NumPy 배열로 변환
    predict_X = tf.convert_to_tensor(arr, dtype=tf.float32) # NumPy 배열을 TensorFlow Tensor로 변환
    
    # 예측
    result = model.predict(predict_X) # 예측
    encoded_labels  = result.argmax() # 가장 확률이 높은 label 찾기
    decoded_labels = le.inverse_transform([encoded_labels]) # 역변환으로 label명 반환

    return decoded_labels
