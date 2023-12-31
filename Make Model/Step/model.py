# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:40:32 2023

@author: admin
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
    def __init__(self, n_categories, embedding_size):
        self.n_categories = n_categories
        self.embedding_size = embedding_size
        
    # 모델 생성
    def make_model(self, Dense_size, Dropout_size):

        self.model = Sequential()

        # 배치 정규화
        self.model.add(BatchNormalization())

        # full-connect layer
        self.model.add(Dense(Dense_size[0], activation='relu', input_shape=(None, self.embedding_size)))
        self.model.add(Dropout(Dropout_size[0]))
        
        for dense_s, drop_p in zip(Dense_size[1:], Dropout_size[1:]):

            self.model.add(Dense(dense_s, activation='relu'))
            self.model.add(Dropout(drop_p))

        # Add Dense layer with softmax activation for classification
        self.model.add(Dense(self.n_categories, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
    
    # 모델 학습
    def model_train(self, train_data, epochs, batch_size):
        
        modelCheck = ModelCheckpoint(
            filepath='Model/best_model.keras', 
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            save_weights_only=True
            ) # validation accuracy를 기준, 가장 좋은 모델만 저장

        # 학습 데이터 가져오기
        X_train,Y_train = train_data

        # Train the model
        self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2,
                            callbacks=modelCheck, verbose=1)

    
    # 모델 평가
    def model_evaluate(self, test_data):

        # 테스트 데이터 가져오기
        X_test, Y_test = test_data
        
        # 가장 최적의 모델 load
        self.model.load_weights('Model/best_model.keras')

        # Evaluate the model
        loss, self.accuracy = self.model.evaluate(X_test, Y_test, verbose=1)
        
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {self.accuracy:.4f}")
     
    # 모델 저장
    def model_save(self, route):
        
        route = route.format(int(self.accuracy*100))
        self.model.save(route)  #학습된 모델 저장