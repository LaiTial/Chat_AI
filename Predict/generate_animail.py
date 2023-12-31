# 동물 답변 생성

import torch
import logging
from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Config

# 로그 조정
logging.getLogger("transformers").setLevel(logging.WARNING)

model_dir = "Model/GPT_Model/animal_GPT"

# 모델 구성 로드
config = GPT2Config.from_pretrained(model_dir)

# 모델 로드
model = GPT2LMHeadModel.from_pretrained(model_dir, config=config)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2',
                                        bos_token='<s>',
                                        eos_token='</s>',
                                        unk_token='<unk>',
                                        pad_token='<pad>',
                                        mask_token='<mask>'
                                        )

def generate_animal_A(sentence, intent, ner):
    
    input_ids = tokenizer.encode(f"<intent>{intent}<ner>{ner}<q>{sentence}<a>")
    gen_ids = model.generate(torch.tensor([input_ids]),
                max_length=128,
                repetition_penalty=2.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                use_cache=True)
    
    decoded_text = tokenizer.decode(gen_ids[0,:].tolist())
        
    # 문장의 끝을 확인하여 해당 끝 이전까지의 부분만 선택
    start_idx = decoded_text.find('<a>')
    end_idx = decoded_text.find('</s>')
    selected_text = decoded_text[start_idx+3:end_idx]
    
    return selected_text

