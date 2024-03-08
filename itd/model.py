import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, GPTNeoXForCausalLM

def load_base_model_and_tokenizer(model_name_or_path):
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True,
                                                max_memory={0: "10GB", 1: "10GB", 2: "0GB", 3: "0GB", 4: "0GB", 5: "0GB", 6: "0GB", 7: "0GB", 8: "0GB", 9: "0GB"})
                                                #max_memory={0: "0GB", 1: "0GB", 2: "0GB", 3: "0GB", 4: "0GB", 5: "0GB", 6: "0GB", 7: "0GB", 8: "10GB", 9: "10GB"})
                                                #max_memory={0: "10GB", 1: "10GB", 2: "10GB", 3: "10GB", 4: "10GB", 5: "10GB", 6: "10GB", 7: "10GB", 8: "10GB", 9: "10GB"})

    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.bos_token
    
    return model, tokenizer

def load_finetuned_model_and_tokenizer(model_name_or_path, finetune_model_path):
    
    config = PeftConfig.from_pretrained(finetune_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True,
                                                #max_memory={0: "5GB", 1: "5GB", 2: "5GB", 3: "5GB", 4: "5GB", 5: "0GB", 6: "0GB", 7: "0GB", 8: "0GB", 9: "0GB"})
                                                #max_memory={0: "0GB", 1: "0GB", 2: "0GB", 3: "0GB", 4: "0GB", 5: "0GB", 6: "20GB", 7: "20GB", 8: "20GB", 9: "0GB"})
                                                max_memory={0: "10GB", 1: "10GB", 2: "10GB", 3: "10GB", 4: "10GB", 5: "10GB", 6: "10GB", 7: "10GB", 8: "10GB", 9: "10GB"})

    model = model.eval()
    
    model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
    
    return model, tokenizer

def load_chatglm(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True, padding_side='left')
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
    
    return model, tokenizer

from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda(8)
    
    return model, tokenizer

def load_gpt_j(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: "0GB", 1: "0GB", 2: "0GB", 3: "0GB", 4: "0GB", 5: "8GB", 6: "8GB", 7: "8GB", 8: "8GB"})
    
    return model, tokenizer