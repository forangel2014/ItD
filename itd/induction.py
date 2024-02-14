import torch
from collections import Counter
from utils.base import StoppingCriteriaList, StopAtSpecificTokenCriteria

def default_post_process(texts):
    texts = [text.replace("<s>", "").replace("</s>", "") for text in texts]
    texts = [text.split("\n")[0] for text in texts]
    #texts = [text.split(".")[0] for text in texts]
    texts = [text.strip(' ".') for text in texts]
    texts = [text + '.' for text in texts]
    return texts

class LLMInductor():
    
    
    def __init__(self, model, tokenizer, max_len=100, device=8, post_process=default_post_process):
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.post_process = post_process

    def induce(self, xs, num_beams):

        # group decoding
        
        input_ids = self.tokenizer(xs, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device)        
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":self.max_len,
            "do_sample":False,#True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":num_beams,
            "num_beams":num_beams,
            "num_beam_groups":"gd-mean"
        }
        model_output = self.model.generate(**generate_input)
        generate_ids = model_output["sequences"]
        generate_ids = generate_ids[:num_beams, input_ids.shape[1]:]
        beam_scores = model_output["scores"][-1][:num_beams]

        # end group decoding

        # re-ranking

        generate_input = {
            "input_ids":generate_ids,
            "max_new_tokens":0,
            "do_sample":False,#True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":1,
            "num_beams":1,
        }
        model_output = self.model.generate(**generate_input)
        logits = model_output["scores"][0]
        log_probs = torch.log_softmax(logits, dim=-1)
        sequence_log_probs = log_probs.gather(2, generate_ids.unsqueeze(-1)).squeeze(-1)
        masks = generate_ids != self.tokenizer.pad_token_id
        sequence_log_probs = sequence_log_probs.multiply(masks)
        sequence_log_probs = torch.sum(sequence_log_probs, dim=-1)

        # end re-ranking
        
        batch_size = len(xs)
        final_scores = -(batch_size-1) * sequence_log_probs + beam_scores

        texts = [self.tokenizer.decode(generate_ids[i]) for i in range(generate_ids.shape[0])]

        texts = self.post_process(texts)

        sorted_texts = [(text, scores) for scores, text in sorted(zip(final_scores.tolist(), texts), reverse=True)]
        
        return sorted_texts
    
    def gd_sample(self, xs, num_beams):

        # group decoding
        
        input_ids = self.tokenizer(xs, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device)        
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":self.max_len,
            "do_sample":True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":num_beams,
            "num_beams":num_beams,
            "num_beam_groups":"gd-mean"
        }
        model_output = self.model.generate(**generate_input)
        generate_ids = model_output["sequences"]
        generate_ids = generate_ids[:num_beams, input_ids.shape[1]:]
        beam_scores = model_output["scores"][-1][:num_beams]

        # end group decoding

        # re-ranking

        generate_input = {
            "input_ids":generate_ids,
            "max_new_tokens":0,
            "do_sample":False,#True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":1,
            "num_beams":1,
        }
        model_output = self.model.generate(**generate_input)
        logits = model_output["scores"][0]
        log_probs = torch.log_softmax(logits, dim=-1)
        sequence_log_probs = log_probs.gather(2, generate_ids.unsqueeze(-1)).squeeze(-1)
        masks = generate_ids != self.tokenizer.pad_token_id
        sequence_log_probs = sequence_log_probs.multiply(masks)
        sequence_log_probs = torch.sum(sequence_log_probs, dim=-1)

        # end re-ranking
        
        batch_size = len(xs)
        final_scores = -(batch_size-1) * sequence_log_probs + beam_scores

        texts = [self.tokenizer.decode(generate_ids[i]) for i in range(generate_ids.shape[0])]

        texts = self.post_process(texts)

        sorted_texts = [(text, scores) for scores, text in sorted(zip(final_scores.tolist(), texts), reverse=True)]
        
        return sorted_texts
    
    def beamsearch(self, xs, num_beams, stop=None):

        if stop:
            token_id_list = [self.tokenizer.encode(stop)[-1]]
            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list))
        else:
            stopping_criteria = None
        
        input_ids = self.tokenizer(xs, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device)        
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":self.max_len,
            "do_sample":False,#True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":num_beams,
            "num_beams":num_beams,
            "stopping_criteria":stopping_criteria,
        }
        model_output = self.model.generate(**generate_input)
        generate_ids = model_output["sequences"]
        generate_ids = generate_ids[:num_beams, input_ids.shape[1]:]
        beam_scores = model_output["scores"][-1][:num_beams]

        final_scores = beam_scores

        texts = [self.tokenizer.decode(generate_ids[i]) for i in range(generate_ids.shape[0])]

        texts = self.post_process(texts)

        sorted_texts = [(text, scores) for scores, text in sorted(zip(final_scores.tolist(), texts), reverse=True)]
        
        return sorted_texts

    def sample(self, xs, num_beams):
        
        input_ids = self.tokenizer(xs, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device)        
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":self.max_len,
            "do_sample":True,
            "top_k":50,
            "top_p":0.5,#0.95
            "temperature":1,#0.3
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":num_beams,
            "num_beams":num_beams,
        }
        model_output = self.model.generate(**generate_input)
        generate_ids = model_output["sequences"]
        generate_ids = generate_ids[:num_beams, input_ids.shape[1]:]
        beam_scores = model_output["scores"][-1][:num_beams]

        final_scores = beam_scores

        texts = [self.tokenizer.decode(generate_ids[i]) for i in range(generate_ids.shape[0])]

        texts = self.post_process(texts)

        sorted_texts = [(text, scores) for scores, text in sorted(zip(final_scores.tolist(), texts), reverse=True)]
        
        return sorted_texts
    
    def self_consistency(self, xs, num_beams, stop=None):
        
        if stop:
            token_id_list = [self.tokenizer.encode(stop)[-1]]
            stopping_criteria = StoppingCriteriaList()
            stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list))
        else:
            stopping_criteria = None
        
        input_ids = self.tokenizer(xs, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device)        
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":self.max_len,
            "do_sample":True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":num_beams,
            "num_beams":num_beams,
            "stopping_criteria":stopping_criteria,
        }
        model_output = self.model.generate(**generate_input)
        generate_ids = model_output["sequences"]
        generate_ids = generate_ids[:num_beams, input_ids.shape[1]:]
        beam_scores = model_output["scores"][-1][:num_beams]

        final_scores = beam_scores

        texts = [self.tokenizer.decode(generate_ids[i]) for i in range(generate_ids.shape[0])]

        texts = self.post_process(texts)
        
        sorted_texts = [(text, scores) for scores, text in sorted(zip(max(final_scores.tolist()), texts), reverse=True)][0]
        
        return sorted_texts
    
    def self_refine(self, xs, num_beams):
        
        input_ids = self.tokenizer(xs, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device)        
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":self.max_len,
            "do_sample":False,#True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":num_beams,
            "num_beams":num_beams,
        }
        model_output = self.model.generate(**generate_input)
        generate_ids = model_output["sequences"]
        generate_ids = generate_ids[:num_beams, input_ids.shape[1]:]
        beam_scores = model_output["scores"][-1][:num_beams]

        texts = [self.tokenizer.decode(generate_ids[i]) for i in range(generate_ids.shape[0])]

        texts = self.post_process(texts)

        feedback_prompt = "\n\nIs the instruction in the story above correct? Can you provide some advice to improve the instruction? Give me a suggestion.\n"
        feedback_input = [xs + ": " + text + feedback_prompt for text in texts]
        feedback_ids = self.tokenizer(feedback_input, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device)        
        
        generate_input = {
            "input_ids":feedback_ids,
            "max_new_tokens":self.max_len,
            "do_sample":False,#True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":num_beams,
            "num_beams":num_beams,
        }
        model_output = self.model.generate(**generate_input)
        generate_ids = model_output["sequences"]
        generate_ids = generate_ids[:num_beams, feedback_ids.shape[1]:]
        beam_scores = model_output["scores"][-1][:num_beams]

        final_scores = beam_scores

        texts = [self.tokenizer.decode(generate_ids[i]) for i in range(generate_ids.shape[0])]

        improve_prompt = "\n\nCan you give one suggestion to improve the induction results above? Don't fix the induction results, just give a suggestion.\n"
        improve_input = [xs + ":\n" + text + improve_prompt for text in texts]
        improve_ids = self.tokenizer(improve_input, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).input_ids.to(self.device)        
        
        generate_input = {
            "input_ids":improve_ids,
            "max_new_tokens":self.max_len,
            "do_sample":False,#True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id,
            "return_dict_in_generate":True,
            "output_scores":True,
            "num_return_sequences":num_beams,
            "num_beams":num_beams,
        }
        model_output = self.model.generate(**generate_input)
        generate_ids = model_output["sequences"]
        generate_ids = generate_ids[:num_beams, improve_ids.shape[1]:]
        beam_scores = model_output["scores"][-1][:num_beams]

        final_scores = beam_scores

        texts = [self.tokenizer.decode(generate_ids[i]) for i in range(generate_ids.shape[0])]

        texts = self.post_process(texts)
        
        sorted_texts = [(text, scores) for scores, text in sorted(zip(max(final_scores.tolist()), texts), reverse=True)][0]
        
        return sorted_texts