# Pet Assistant Bot
        서경대학교 소프트웨어학과 졸업작품
        SW 인재교육 참여기업 오브젠에서 제안 
        자연어 처리(NLP)를 사용한 챗봇 만들기.
![temp_4](https://github.com/LaiTial/Chat/assets/39575609/060cc7f1-21f1-4a38-b8a8-103519465dc4)

## 프로젝트명
#### PET Assistant Bot
![temp_4](https://github.com/LaiTial/Chat/assets/39575609/ee5a25ef-329a-41c0-a6f8-e92039a09f41)

## 프로젝트 소개
#### 반려동물을 키우는 사람들을 위한 도움을 주는 챗봇의 Flask Server & AI model
![temp_4](https://github.com/LaiTial/Chat/assets/39575609/46f3d7b3-262b-4fb3-a02b-7d591d0695b2)

## 개발 기간
2023.10.23 ~ 2023.12.27
![temp_4](https://github.com/LaiTial/Chat/assets/39575609/569914a2-4999-4d71-a95e-cb8ee17e45a4)

## 구현언어
- sentence-bert, transformers, python, tensorflow 등 사용
![temp_4](https://github.com/LaiTial/Chat/assets/39575609/46f3d7b3-262b-4fb3-a02b-7d591d0695b2)

 ## 개발 기능
- 챗봇 기능
![temp_4](https://github.com/LaiTial/Chat/assets/39575609/d21fa01a-0168-44b2-b2bf-b47ab5a08ae3)

## 요구 사항
- openai와 crawling을 활용한 질문-답변 수집
- 데이터 전처리
- sentence-bert를 사용한 자연어 처리 & 임베딩
- tensorflow를 활용한 분류 모델
- 코사인 유사도로 유사도 계산 & DB 탐색
- kogpt-2를 파인튜닝한 일상&동물 답변 생

![temp_4](https://github.com/LaiTial/Chat/assets/39575609/5c30fbe0-1a5d-4e2f-b54f-3bef148d16ef)
                                      
## 딥러닝 설계계
### 1. 챗봇 응답 흐름도
<img src="https://github.com/LaiTial/Chat_AI/assets/39575609/99115bcc-d568-456f-a7d5-5eae62d4c7d0" alt="이미지 설명" width="900">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">

### 2. 질문 분류 모델
<img src="https://github.com/LaiTial/Chat_AI/assets/39575609/5c01bd60-31b1-47f4-b048-6dc7913f77d7" alt="blank" width="800">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">

### 3. 종 분류 모델
<img src="https://github.com/LaiTial/Chat_AI/assets/39575609/53a5c08d-c816-444e-aeb2-b986cb9a1713" alt="이미지 설명" width="800">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">

### 4. 의도 분류 모델
<img src="https://github.com/LaiTial/Chat_AI/assets/39575609/6c1b4df2-555f-4f04-8017-982c074293a2" alt="이미지 설명" width="800">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">

### 5. 유사도 판단 모델
<img src="https://github.com/LaiTial/Chat_AI/assets/39575609/561c569e-e659-4c13-bbcc-7e5fcb3d07db" alt="이미지 설명" width="800">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">

### 6. 유사도 계산
<img src="https://github.com/LaiTial/Chat_AI/assets/39575609/6fc9443c-2bf8-4991-b146-225a66ca920d" alt="이미지 설명" width="900">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">

### 7. 일상 답변 생성 모델
<img src="https://github.com/LaiTial/Chat_AI/assets/39575609/ad3bb7a6-b82e-476c-81c0-44f08ebbd1a1" alt="이미지 설명" width="700">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">

### 8. 동물 답변 생성 모델
<img src="https://github.com/LaiTial/Chat_AI/assets/39575609/96e1ff0f-36b5-454e-bea3-fc3cc0744f69" alt="이미지 설명" width="700">
<img src="https://github.com/LaiTial/Chat/assets/39575609/e92ac0ca-1c2a-4d2c-bdc7-7285c7ab9ebd" alt="blank" width="700">

## DB
### 질문-답변
![image](https://github.com/LaiTial/Chat_AI/assets/39575609/64ee4618-9ed1-455f-b5ad-04a5f57aba68)











