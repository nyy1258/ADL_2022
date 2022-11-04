python preprocess.py --do_train --do_predict

python run_mc.py \
--model_name_or_path "hfl/chinese-xlnet-base" \
--output_dir ./multi_choice_model/ \
--overwrite_output_dir \
--per_gpu_train_batch_size 1 \
--save_steps 10000 \
--gradient_accumulation_steps 2 \
--max_seq_length 512 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--do_train \
--do_eval \
--train_file './cache/pre_train.json' \
--validation_file './cache/pre_valid.json' \
--do_predict \
--test_file './cache/pre_test.json'


python run_qa.py \
--model_name_or_path "hfl/chinese-xlnet-base" \
--output_dir ./question_answer_model/ \
--overwrite_output_dir \
--per_gpu_train_batch_size 1 \
--save_steps 10000 \
--gradient_accumulation_steps 2 \
--max_seq_length 512 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--do_train \
--do_eval \
--train_file './cache/pre_train.json' \
--validation_file './cache/pre_valid.json' \
--do_predict \
--test_file './cache/pre_test.json'


python postprocess.py --file ./question_answer_model/predict_predictions.json --out ./pred.csv