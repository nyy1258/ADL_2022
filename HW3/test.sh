python run_summarization.py \
--model_name_or_path google/mt5-small \
--output_dir ./summariztion \
--do_train \
--do_eval \
--train_file ./data/train.jsonl \
--validation_file ./data/public.jsonl \
--summary_column title \
--text_column maintext \
--source_prefix "summarize: " \
--per_gpu_train_batch_size 1  \
--gradient_accumulation_steps 2 \
--learning_rate 1e-3 \
--predict_with_generate \
--num_train_epochs 20 \
--overwrite_output_dir \
--do_predict \
--test_file ./data/public.jsonl 


