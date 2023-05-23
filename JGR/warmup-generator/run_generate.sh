# for training set use sampling
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name cnndm \
--split train \
--use_tokenized_data True \
--do_predict \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_do_sample True \
--generation_num_beams 1 \
--generation_num_return_sequences 16 \
--model_name_or_path saves/bart-large-cnndm \
--config_name saves/bart-large-cnndm \
--tokenizer_name saves/bart-large-cnndm \
--max_source_length 1020 --max_target_length 100 --disable_tqdm False \
--output_dir predict --save_name cnndm-large

# for dev and test, use beam search
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name cnndm \
--split dev \
--use_tokenized_data True \
--do_predict \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_do_sample False \
--generation_num_beams 16 \
--generation_num_return_sequences 16 \
--model_name_or_path saves/bart-large-cnndm \
--config_name saves/bart-large-cnndm \
--tokenizer_name saves/bart-large-cnndm \
--max_source_length 1020 --max_target_length 100 --disable_tqdm False \
--output_dir predict --save_name cnndm-large

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name cnndm \
--split test \
--use_tokenized_data True \
--do_predict \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_do_sample False \
--generation_num_beams 16 \
--generation_num_return_sequences 16 \
--model_name_or_path saves/bart-large-cnndm \
--config_name saves/bart-large-cnndm \
--tokenizer_name saves/bart-large-cnndm \
--max_source_length 1020 --max_target_length 100 --disable_tqdm False \
--output_dir predict --save_name cnndm-large



# # for training set use sampling
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name samsum \
# --split train \
# --use_tokenized_data True \
# --do_predict \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
# --generation_do_sample True \
# --generation_num_beams 1 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-samsum \
# --config_name saves/bart-large-samsum \
# --tokenizer_name saves/bart-large-samsum \
# --max_source_length 1020 --max_target_length 100 --disable_tqdm False \
# --output_dir predict --save_name samsum-large

# # for dev and test, use beam search
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name samsum \
# --split dev \
# --use_tokenized_data True \
# --do_predict \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
# --generation_do_sample False \
# --generation_num_beams 16 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-samsum \
# --config_name saves/bart-large-samsum \
# --tokenizer_name saves/bart-large-samsum \
# --max_source_length 1020 --max_target_length 100 --disable_tqdm False \
# --output_dir predict --save_name samsum-large

# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name samsum \
# --split test \
# --use_tokenized_data True \
# --do_predict \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
# --generation_do_sample False \
# --generation_num_beams 16 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-samsum \
# --config_name saves/bart-large-samsum \
# --tokenizer_name saves/bart-large-samsum \
# --max_source_length 1020 --max_target_length 100 --disable_tqdm False \
# --output_dir predict --save_name samsum-large


# # for training set use sampling
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name personachat \
# --split train \
# --use_tokenized_data False \
# --do_predict --per_device_eval_batch_size 8 \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 70 \
# --generation_do_sample True \
# --generation_num_beams 1 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-personachat \
# --config_name saves/bart-large-personachat \
# --tokenizer_name saves/bart-large-personachat \
# --max_source_length 700 --max_target_length 70 --disable_tqdm False \
# --output_dir predict --save_name personachat-large

# # for dev and test, use beam search
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name personachat \
# --split dev \
# --use_tokenized_data False \
# --do_predict --per_device_eval_batch_size 8 \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 70 \
# --generation_do_sample False \
# --generation_num_beams 16 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-personachat \
# --config_name saves/bart-large-personachat \
# --tokenizer_name saves/bart-large-personachat \
# --max_source_length 700 --max_target_length 70 --disable_tqdm False \
# --output_dir predict --save_name personachat-large

# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name personachat \
# --split test \
# --use_tokenized_data False \
# --do_predict --per_device_eval_batch_size 8 \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 70 \
# --generation_do_sample False \
# --generation_num_beams 16 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-personachat \
# --config_name saves/bart-large-personachat \
# --tokenizer_name saves/bart-large-personachat \
# --max_source_length 700 --max_target_length 70 --disable_tqdm False \
# --output_dir predict --save_name personachat-large




# for training set use sampling
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name squadqg \
# --split train \
# --use_tokenized_data False \
# --do_predict --per_device_eval_batch_size 8 \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 65 \
# --generation_do_sample True \
# --generation_num_beams 1 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --config_name saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --tokenizer_name saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --max_source_length 700 --max_target_length 65 --disable_tqdm False \
# --output_dir predict --save_name squadqg-large-new

# for dev and test, use beam search
# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name squadqg \
# --split dev \
# --use_tokenized_data False \
# --do_predict --per_device_eval_batch_size 8 \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 65 \
# --generation_do_sample False \
# --generation_num_beams 16 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --config_name saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --tokenizer_name saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --max_source_length 700 --max_target_length 65 --disable_tqdm False \
# --output_dir predict --save_name squadqg-large-new

# python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
# --dataset_name squadqg \
# --split test \
# --use_tokenized_data False \
# --do_predict --per_device_eval_batch_size 8 \
# --dataloader_pin_memory True --predict_with_generate True --generation_max_length 65 \
# --generation_do_sample False \
# --generation_num_beams 16 \
# --generation_num_return_sequences 16 \
# --model_name_or_path saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --config_name saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --tokenizer_name saves/bart-large-squadqg-bs64-lr1e-5-eval_in_test \
# --max_source_length 700 --max_target_length 65 --disable_tqdm False \
# --output_dir predict --save_name squadqg-large-new
