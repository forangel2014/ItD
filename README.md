## Deductive Data Generation
```
python run_insin.py --exp_dir ./exp/insin/sample --base_model <model_path or chatgpt> --mode io-sample --run_induction
python deduction_insin.py --exp_dir ./exp/insin/<your exp name> --base_model <model_path or chatgpt> --mode <io or gd> --load_from_induced ./exp/insin/sample/induction-out/io-sample
```

## fine-tuning

```
cd finetune
bash finetune_lora.sh
```

## Induction
```
python run_insin.py --exp_dir ./exp/insin/<your exp name> --base_model <model_path or chatgpt> --mode <io or gd> --run_induction --run_deduction
```