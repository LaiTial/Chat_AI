"""06. 질문-답변 생성.ipynb
    
[처음부터 학습]

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

[이미 학습된 모델을 재학습]

# 모델 구성, 모델 로드
config = GPT2Config.from_pretrained(input_dir)
model = GPT2LMHeadModel.from_pretrained(input_dir, config=config)
"""

import pandas as pd
import logging
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Config, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from datasets import Dataset
from torch.utils.data import random_split
        
def read_data():
    route = 'Data/ChatbotData.csv'
    csv = pd.read_csv(route, encoding = 'utf-8')
    csv = csv.drop(columns='label')
    
    return csv

# 전처리 함수 정의
def preprocess_qa(QALine):

  Q, A = QALine

  return f"<s>{Q}<sep>{A}</s>"

# tqdm 진행 막대 바 표시
class TQDMCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        steps_per_epoch = state.max_steps/state.num_train_epochs
        self.epoch_iterator = tqdm(total=steps_per_epoch, desc=f"Epoch {state.epoch}", position=0, leave=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_iterator.close()

    def on_step_end(self, args, state, control, **kwargs):
        self.epoch_iterator.update(1)

# perplexity 계산
def compute_metrics(p):
    
    # 평가 결과에서 predictions와 label_ids를 가져와 perplexity 계산
    predictions = torch.softmax(torch.tensor(p.predictions), dim=-1)
    label_ids = torch.tensor(p.label_ids)

    # 모델의 출력 크기에 맞게 레이블 크기를 맞춤
    label_ids = label_ids[:, :predictions.size(1)]

    # loss를 직접 계산하여 perplexity를 구함
    loss = torch.nn.functional.cross_entropy(predictions.permute(0, 2, 1), label_ids)
    perplexity = torch.exp(loss).item()

    return {"perplexity": perplexity}

# 챗봇 데이터 로드
csv = read_data()
    
# 로그 조정
logging.getLogger("transformers").setLevel(logging.WARNING)

input_dir = "GPT_Model/Daily"

# 모델 구성 로드
config = GPT2Config.from_pretrained(input_dir)

# 모델 로드
model = GPT2LMHeadModel.from_pretrained(input_dir, config=config)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2',
                                        bos_token='<s>',
                                        eos_token='</s>',
                                        unk_token='<unk>',
                                        pad_token='<pad>',
                                        mask_token='<mask>'
                                        )

# 전처리된 데이터셋 생성
processed_data = [preprocess_qa(line) for line in csv.values]
train_dataset = tokenizer(processed_data, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Dataset 생성
train_dataset = Dataset.from_dict({'input_ids': train_dataset['input_ids'], 'attention_mask': train_dataset['attention_mask']})

# 전체 데이터셋 크기
total_size = len(train_dataset)

# 평가 데이터셋 크기 계산 (20%)
eval_size = int(0.2 * total_size)

# 훈련 데이터셋과 평가 데이터셋으로 분할
train_size = total_size - eval_size
train_dataset, eval_dataset = random_split(train_dataset, [train_size, eval_size])


# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    mlm_probability=0.20
)

# 학습 인수 정의
batch_size=1024
epochs=1
output_dir = "Model/GPT_Model/Daily"

# 파인튜닝을 위한 훈련 인자 설정
training_args=TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    log_level="info", # 로그 레벨
    logging_strategy = "steps",
    logging_steps=1,
    save_strategy='epoch',
    evaluation_strategy='epoch',  # 한 epoch 마다 평가
    eval_steps=1,  # 하나의 epoch마다
    load_best_model_at_end=True,
    metric_for_best_model="perplexity",  # perplexity를 평가 지표로 설정
    greater_is_better=False, # 평가지표가 낮게 모델을 조정
    lr_scheduler_type="constant_with_warmup",  # lr 스케줄러 지정
    warmup_ratio=0.1, # 전체 훈련 스텝에 대한 초기 학습률의 비율
    warmup_steps=0, # 웜업 스텝의 수
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, 
                                                early_stopping_threshold=20)  # 평가지표가 20 이상 감소해야 조기 종료 조건을 충족

# 트레이너 생성 및 파인튜닝 시작
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 평가 데이터셋을 지정,
    compute_metrics=compute_metrics,  # perplexity 계산 함수 추가
    callbacks=[early_stopping_callback, TQDMCallback()],
)


# 훈련 전에 CUDA 캐시 비우기
torch.cuda.empty_cache()

# 모델 훈련 및 평가
trainer.train() 

# 최적의 모델을 평가 및 로드
results = trainer.evaluate()
print("Loss", results["eval_loss"])
print("Perplexity:", results["eval_perplexity"])

# 모델 저장
trainer.save_model()

# 파인튜닝이 완료된 후 메모리를 해제
del trainer
torch.cuda.empty_cache()