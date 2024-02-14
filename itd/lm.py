import os
import openai
import numpy as np
import torch
import math
from collections import Counter
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from itd.model import load_base_model_and_tokenizer
from utils.base import StoppingCriteriaList, StopAtSpecificTokenCriteria

class LLM():
    
    def __init__(self, model_name_or_path, max_token, device, finetuned_model_id=None):
        
        self.device = device
        self.max_token = max_token
        self.finetuned_model_id = finetuned_model_id
        if type(model_name_or_path) == str:
            if model_name_or_path in ['gpt3', 'chatgpt', 'gpt4']:
                self.model = model_name_or_path
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                self.model, self.tokenizer = load_base_model_and_tokenizer(model_name_or_path)
                self.ending_idx = self.tokenizer.eos_token_id
                self.padding_idx = self.tokenizer.pad_token_id
        else:
            self.model, self.tokenizer = model_name_or_path
            self.ending_idx = self.tokenizer.eos_token_id
            self.padding_idx = self.tokenizer.pad_token_id

    def __call__(self, prompt, stop=None, temperature=0, do_sample=False, top_p=0):
        if self.model == 'gpt3':
            response = self.gpt3(prompt, stop)
        elif self.model == 'chatgpt':
            response = self.chatgpt(prompt, stop, temperature, top_p)
        elif self.model == 'gpt4':
            response = self.gpt4(prompt, stop, temperature, top_p)
        else:
            response = self.local(prompt, stop, do_sample=do_sample)
        #response = post_process(response)
        return response
    
    def local(self, prompt, stop, do_sample):
        
        if stop:
            token_id_list = [self.tokenizer.encode(stop)[-1]]
            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list))
        else:
            stopping_criteria = None
            
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        response = self.model.generate(inputs, 
                                        max_length=inputs.shape[1]+self.max_token, 
                                        early_stopping=True,
                                        eos_token_id=self.ending_idx,
                                        pad_token_id=self.padding_idx,
                                        do_sample=do_sample,
                                        stopping_criteria=stopping_criteria
                                        )
        
        response = response[0][inputs.shape[1]:]
        response_text = self.tokenizer.decode(response).strip('\n')
        return response_text
    
    def gpt3(self, prompt, stop=["\n"]):
        response = openai.Completion.create(
                        model="davinci-002",
                        prompt=prompt,
                        temperature=0,
                        max_tokens=100,
                        top_p=0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        #stop=stop
                    )
        return response["choices"][0]["text"]
    
    def chatgpt(self, prompt, stop, temperature=0, top_p=0):
        messages = [{"role": "user", "content": prompt}]
        model_id = self.finetuned_model_id if self.finetuned_model_id else "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(
                model=model_id,#"gpt-3.5-turbo", 
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_token,
                top_p=top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop
            )
        return response["choices"][0]["message"]["content"]

    def gpt4(self, prompt, stop, temperature=0, top_p=0):
        messages = [{"role": "user", "content": prompt}]
        model_id = self.finetuned_model_id if self.finetuned_model_id else "gpt-4"
        response = openai.ChatCompletion.create(
                model=model_id,#"gpt-3.5-turbo", 
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_token,
                top_p=top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=stop
            )
        return response["choices"][0]["message"]["content"]
    
    
    def lm_loss(self, prompt, target):
        prompt_inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        total_inputs = self.tokenizer.encode(prompt+target, return_tensors='pt').to(self.device)
        target_inputs = self.tokenizer.encode(target, return_tensors='pt').to(self.device)

        # 直接解码target文本的损失
        with torch.no_grad():
            target_loss = self.model(target_inputs, labels=target_inputs)[0]
            # logits = self.model(total_inputs)[0]
            # loss_fct = torch.CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels)
            
            prompt_loss = self.model(prompt_inputs, labels=prompt_inputs)[0]
            total_loss = self.model(total_inputs, labels=total_inputs)[0]
            
            target_loss_given_prompt = (total_loss*(total_inputs.shape[1]-1)-prompt_loss*(prompt_inputs.shape[1]-1))/(target_inputs.shape[1]-1)
        
        return target_loss.tolist(), target_loss_given_prompt.tolist()

def post_process(response):
    processed_response = response.rstrip('\n.')
    return processed_response
