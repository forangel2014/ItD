## 关于chatgpt微调实验
1. chatgpt归纳（采样先验f）
```
python run_insin.py --exp_dir ./exp/insin/chatgpt --base_model chatgpt --mode io-sample --run_induction
```

2. chatgpt演绎（基于f，生成x,y）
```
python deduction_insin.py --exp_dir ./exp/insin/induced1_chatgpt --base_model chatgpt --mode io --load_from_induced ./exp/insin/chatgpt/induction_out/io-sample
```

3. chatgpt微调（基于演绎得到的f,x,y）


4. 测试微调后的模型
```
python run_insin.py --exp_dir ./exp/insin/induced1_chatgpt --base_model chatgpt --mode io --run_induction --run_deduction
```