# cnndm
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_train.py \
--dataset_name cnndm \
--report_to none \
--do_train True --do_eval True --do_predict True \
--evaluation_strategy epoch  --logging_strategy epoch --save_strategy epoch \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--num_train_epochs 5 \
--metric_for_best_model rouge2 \
--generation_num_beams 4 --generation_max_length 100 \
--model_name_or_path bart-large \
--config_name bart-large \
--tokenizer_name bart-large \
--max_source_length 400 --max_target_length 109 \
--output_dir saves/bart-large-cnndm  --overwrite_output_dir

# samsum
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_train.py \
--dataset_name samsum \
--report_to none \
--do_train True --do_eval True --do_predict True \
--evaluation_strategy epoch  --logging_strategy epoch --save_strategy epoch \
--per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
--num_train_epochs 3000 \
--metric_for_best_model rouge2 \
--generation_num_beams 4 --generation_max_length 100 \
--model_name_or_path bart-large \
--config_name bart-large \
--tokenizer_name bart-large \
--max_source_length 400 --max_target_length 109 \
--output_dir saves/bart-large-samsum  --overwrite_output_dir

# squadqg
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_train.py \
--dataset_name squadqg \
--report_to none \
--do_train True --do_eval True --do_predict True \
--evaluation_strategy epoch  --logging_strategy epoch --save_strategy epoch \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--num_train_epochs 5 \
--metric_for_best_model rougeL \
--generation_num_beams 4 --generation_max_length 65 \
--model_name_or_path bart-large \
--config_name bart-large \
--tokenizer_name bart-large \
--max_source_length 600 --max_target_length 65 \
--output_dir saves/bart-large-squadqg  --overwrite_output_dir

# personachat
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_train.py \
--dataset_name personachat \
--report_to none \
--do_train True --do_eval True --do_predict True \
--evaluation_strategy epoch  --logging_strategy epoch --save_strategy epoch \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--num_train_epochs 5 \
--metric_for_best_model rougeL \
--generation_num_beams 4 --generation_max_length 70 \
--model_name_or_path bart-large \
--config_name bart-large \
--tokenizer_name bart-large \
--max_source_length 700 --max_target_length 70 \
--output_dir saves/bart-large-personachat  --overwrite_output_dir