python -m torch.distributed.launch --nproc_per_node 8  run_train.py --overwrite_output_dir \
    --task_name sum --dataset_name cnndm \
    --train_data_path data/cnndm \
    --dev_data_path data/cnndm \
    --test_data_path data/cnndm \
    --load_tokenized_data False \
    --generator_num_cand_generated 8 --generator_num_cand_picked 8 \
    --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
    --do_train True --do_eval False --do_predict False --prediction_loss_only False \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --generator_learning_rate 5e-5 --reranker_learning_rate 1e-5 \
    --num_train_epochs 3 \
    --evaluation_strategy steps --eval_steps 1000 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 1000 --save_total_limit 20 \
    --iteration_steps 1000 --iteration_reranker_steps 500 \
    --load_best_model_at_end True \
    --metric_for_best_model generator_eval_rouge1 --greater_is_better True \
    --reranker_model_name_or_path warmup-ranker/saves/roberta-large-cnndm \
    --generator_model_name_or_path warmup-generator/saves/bart-large-cnndm \
    --output_dir saves/JGR-large-cnndm \
    --generator_max_source_length 1020 --reranker_max_source_length 400 --generator_max_target_length 109 --reranker_max_target_length 109 \
    --cache_data \
    --disable_tqdm False 



python -m torch.distributed.launch --nproc_per_node 8  run_train.py --overwrite_output_dir \
    --task_name sum --dataset_name samsum \
    --train_data_path data/samsum \
    --dev_data_path data/samsum \
    --test_data_path data/samsum \
    --load_tokenized_data False \
    --generator_num_cand_generated 8 --generator_num_cand_picked 8 \
    --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
    --do_train True --do_eval False --do_predict False --prediction_loss_only False \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --generator_learning_rate 1e-5 --reranker_learning_rate 5e-6 \
    --num_train_epochs 10 \
    --evaluation_strategy steps --eval_steps 462 \
    --logging_strategy steps --logging_steps 231 \
    --save_strategy steps --save_steps 462 --save_total_limit 20 \
    --iteration_steps 462 --iteration_reranker_steps 231 \
    --load_best_model_at_end True \
    --metric_for_best_model generator_eval_rouge1 --greater_is_better True \
    --reranker_model_name_or_path warmup-ranker/saves/roberta-large-samsum \
    --generator_model_name_or_path warmup-generator/saves/bart-large-samsum \
    --output_dir saves/JGR-large-samsum \
    --generator_max_source_length 1020 --reranker_max_source_length 400 --generator_max_target_length 109 --reranker_max_target_length 109 \
    --cache_data \
    --disable_tqdm False 


python -m torch.distributed.launch --nproc_per_node 8  run_train.py --overwrite_output_dir \
    --task_name qg --dataset_name squadqg \
    --train_data_path data/squadqg \
    --dev_data_path data/squadqg \
    --test_data_path data/squadqg \
    --load_tokenized_data False \
    --generator_num_cand_generated 8 --generator_num_cand_picked 8 \
    --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
    --do_train True --do_eval False --do_predict False --prediction_loss_only False \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --generator_learning_rate 5e-5 --reranker_learning_rate 1e-5 \
    --num_train_epochs 3 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 250 --save_total_limit 20 \
    --iteration_steps 500 --iteration_reranker_steps 250 \
    --load_best_model_at_end True \
    --metric_for_best_model generator_eval_rougeL --greater_is_better True \
    --reranker_model_name_or_path warmup-ranker/saves/roberta-large-squadqg \
    --generator_model_name_or_path warmup-generator/saves/bart-large-squadqg \
    --output_dir saves/JGR-large-squadqg \
    --generator_max_source_length 600 --reranker_max_source_length 435 --generator_max_target_length 65 --reranker_max_target_length 65 \
    --cache_data \
    --disable_tqdm False 


python -m torch.distributed.launch --nproc_per_node 8  run_train.py --overwrite_output_dir \
    --task_name dialog --dataset_name personachat \
    --train_data_path data/personachat \
    --dev_data_path data/personachat \
    --test_data_path data/personachat \
    --load_tokenized_data False \
    --generator_num_cand_generated 8 --generator_num_cand_picked 8 \
    --num_cand_generated 16 --num_cand_picked 3 --candidate_pick_strategy bottom \
    --do_train True --do_eval False --do_predict False --prediction_loss_only False \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --generator_learning_rate 5e-5 --reranker_learning_rate 1e-5 \
    --num_train_epochs 3 \
    --evaluation_strategy steps --eval_steps 1000 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 1000 --save_total_limit 20 \
    --iteration_steps 1000 --iteration_reranker_steps 500 \
    --load_best_model_at_end True \
    --metric_for_best_model generator_eval_rouge1 --greater_is_better True \
    --reranker_model_name_or_path warmup-ranker/saves/roberta-large-personachat \
    --generator_model_name_or_path warmup-generator/saves/bart-large-personachat \
    --output_dir saves/JGR-large-personachat \
    --generator_max_source_length 550 --reranker_max_source_length 430 --generator_max_target_length 70 --reranker_max_target_length 70 \
    --cache_data \
    --disable_tqdm False 