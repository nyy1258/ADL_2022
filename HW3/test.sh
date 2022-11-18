
CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--model_name_or_path google/mt5-small \
--output_dir ./summariztion \
--do_train \
--do_eval \
--train_file ./data/train.jsonl \
--validation_file ./data/public.jsonl \
--output_file ./pred.jsonl \
--summary_column title \
--text_column maintext \
--num_train_epochs 5 \
--source_prefix "summarize: " \
--evaluation_strategy steps \
--eval_steps 2000 \
--logging_steps 2000 \
--save_steps 2000 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-3 \
--warmup_ratio 0.1 \
--overwrite_output_dir \
--predict_with_generate \
--metric_for_best_model rouge-1 \
--metric_for_best_model rouge-2 \
--metric_for_best_model rouge-l 



## predict

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./pred.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./ \
--per_device_eval_batch_size 4 \
--num_beams 5


CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/greedy.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/greedy \
--per_device_eval_batch_size 4 \
--num_beams 1


## evaluate

python eval.py -r ./data/public.jsonl -s pred.jsonl

python eval.py -r ./data/public.jsonl -s num_beam5.jsonl

python eval.py -r ./data/public.jsonl -s ./strategy/greedy.jsonl >> ./greedy.txt



## reference

## test num beam = 1
CUDA_VISIBLE_DEVICES=0,1 python run_summarization.py \
    --model_name_or_path ./summarization/ \
    --do_predict \
    --test_file ./data/public.jsonl \
    --source_prefix "summarize: " \
    --text_column maintext \
    --output_dir ./preds/ \
    --overwrite_output_dir \
    --predict_with_generate \
    --fp16 yes \
    --num_beams 1

#python postprocess.py --input ./data/public.jsonl --output output_beam1.jsonl

python3 combine.py  ./data/public.jsonl  output_beam1.jsonl

python3 eval.py -r ./data/public.jsonl -s ./output_beam1.jsonl