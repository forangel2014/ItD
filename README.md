## Deductive Data Generation
```
python run_insin.py --exp_dir ./exp/insin/sample --base_model <model_path or chatgpt> --mode io-sample --run_induction
python deduction_insin.py --exp_dir ./exp/insin/<your exp name> --base_model <model_path or chatgpt> --mode <io or gd> --load_from_induced ./exp/insin/sample/induction-out/io-sample
```

## fine-tuning with LoRA

```
cd finetune
bash finetune_lora.sh
```

## fine-tuning with OpenAI API

```
python prepare_chatgpt_data.py --exp_dir ./exp/insin/<your exp name>
```

Then upload the ./exp/insin/<your exp name>/chatgpt_train.jsonl file onto the OpenAI website and fine-tune ChatGPT

## Induction
```
python run_insin.py --exp_dir ./exp/insin/<your exp name> --base_model <model_path or chatgpt> --finetuned_model ckpt/checkpoint-<step> --mode <io or gd> --run_induction --run_deduction
```