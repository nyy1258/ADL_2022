CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/beam_10.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/beam_10 \
--per_device_eval_batch_size 4 \
--num_beams 10

python eval.py -r ./data/public.jsonl -s ./strategy/beam_10.jsonl >> ./result/beam_10.txt


CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/top_k10.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/top_k10 \
--per_device_eval_batch_size 4 \
--do_sample True \
--top_k 10

python eval.py -r ./data/public.jsonl -s ./strategy/top_k10.jsonl >> ./result/top_k10.txt

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/top_k20.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/top_k20 \
--per_device_eval_batch_size 4 \
--do_sample True \
--top_k 20

python eval.py -r ./data/public.jsonl -s ./strategy/top_k20.jsonl >> ./result/top_k20.txt

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/top_p0.5.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/top_p_0.5 \
--per_device_eval_batch_size 4 \
--do_sample True \
--top_p 0.5

python eval.py -r ./data/public.jsonl -s ./strategy/top_p0.5.jsonl >> ./result/top_p0.5.txt

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/top_p_0.8.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/top_p_0.8 \
--per_device_eval_batch_size 4 \
--do_sample True \
--top_p 0.8

python eval.py -r ./data/public.jsonl -s ./strategy/top_p_0.8.jsonl >> ./result/top_p_0.8.txt

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/temperature_0.5.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/temperature_0.5 \
--per_device_eval_batch_size 4 \
--do_sample True \
--temperature 0.5

python eval.py -r ./data/public.jsonl -s ./strategy/temperature_0.5.jsonl >> ./result/temperature_0.5.txt

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/temperature_0.8.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/temperature_0.8 \
--per_device_eval_batch_size 4 \
--do_sample True \
--temperature 0.8

python eval.py -r ./data/public.jsonl -s ./strategy/temperature_0.8.jsonl >> ./result/temperature_0.8.txt


CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
--do_predict \
--model_name_or_path  ./summariztion  \
--test_file ./data/public.jsonl  \
--output_file ./strategy/non_greedy.jsonl \
--predict_with_generate \
--text_column maintext \
--output_dir ./strategy/non_greedy \
--per_device_eval_batch_size 4 \


python eval.py -r ./data/public.jsonl -s ./strategy/non_greedy.jsonl >> ./result/non_greedy.txt

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

python eval.py -r ./data/public.jsonl -s ./strategy/greedy.jsonl >> ./result/greedy.txt