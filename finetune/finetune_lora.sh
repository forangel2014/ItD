#exp_dir=../exp/insin/70b_induced1
#finetuned_model=None #../exp/insin/induced1/ckpt/checkpoint-520

exp_dir=../exp/listfunc/induced_mixtral
finetuned_model=None #../exp/listfunc/induced1/ckpt/checkpoint-240

#base_model=/netcache/huggingface/Llama-2-70b-chat-hf 
#base_model=/netcache/huggingface/Llama-2-13b-chat-hf 
#base_model=../../llama2-cn/llama-2-7b-chat
base_model=/netcache/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1

if [ ! -d ${exp_dir} ];then  
    mkdir ${exp_dir}
fi

train_files=${exp_dir}/train.csv
valid_files=${exp_dir}/valid.csv
cp ./finetune_lora.sh ${exp_dir}

deepspeed \
    --include localhost:0,1,2,3,4,5,6,7,8,9 \
    --master_port 10170 \
    finetune_clm_lora.py \
    --model_name_or_path ${base_model}+++${finetuned_model} \
    --train_files ${train_files} \
    --validation_files  ${valid_files} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${exp_dir}/ckpt \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --warmup_steps 400 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${exp_dir}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 20 \
    --eval_steps 20 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --fp16 \
    --gradient_checkpointing \
    --fp16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${exp_dir}/train.log
    
    #bf16
    #bf16_full_eval

    # --resume_from_checkpoint ${exp_dir}/checkpoint-20400 \
