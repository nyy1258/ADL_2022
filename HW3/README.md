1.
Only T5 models `t5-small`, `t5-base`, `t5-large`, `t5-3b` and `t5-11b` must use an additional argument: `--source_prefix "summarize: "`.


2.
`--train_file`, `--validation_file`, `--text_column` and `--summary_column` to match your setup:

--text_column maintext 
--summary_column title 

3.
per_gpu_train_batch_size



python run_summarization.py \
--model_name_or_path google/mt5-small \
--output_dir ./summariztion \
--do_train \
--do_eval \
--train_file ./data/train.jsonl \
--validation_file ./data/public.jsonl \
--output_file ./pred.jsonl \
--summary_column title \
--text_column maintext \
--source_prefix "summarize: " \
--per_gpu_train_batch_size 1  \
--gradient_accumulation_steps 2 \
--save_steps 20000 \
--learning_rate 1e-3 \
--num_train_epochs 3 \
--overwrite_output_dir \
--predict_with_generate \
--do_predict \

--test_file ./data/public.jsonl 




other:
1.
 --evaluation_strategy steps --eval_steps 20000 --learning_rate 4e-5

2.
Training:
 
--eval_accumulation_steps=128 \
 --learning_rate 1e-3 \
  --warmup_ratio 0.1 \





output_file

    do_top_k
    do_top_p
    do_temp
    top_k
    top_p
    temperature
