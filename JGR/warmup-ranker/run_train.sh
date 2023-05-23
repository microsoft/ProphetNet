python -m torch.distributed.launch --nproc_per_node 8  run_reranker.py --overwrite_output_dir \
    --task_name sum --dataset_name cnndm \
    --train_data_path data/cnndm/train/gen_from_16 \
    --dev_data_path data/cnndm/train/gen_from_16 \
    --test_data_path data/cnndm/train/gen_from_16 \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data True \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.2 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 20 \
    --load_best_model_at_end True \
    --metric_for_best_model rouge1 --greater_is_better True \
    --model_name_or_path roberta-large \
    --output_dir saves/roberta-large-cnndm \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 400 --max_candidate_length 109 \
    --cache_data --disable_tqdm False \

python -m torch.distributed.launch --nproc_per_node 8  run_reranker.py --overwrite_output_dir \
    --task_name sum --dataset_name samsum \
    --train_data_path data/samsum/train/gen_from_16 \
    --dev_data_path data/samsum/train/gen_from_16 \
    --test_data_path data/samsum/train/gen_from_16 \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data True \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --warmup_steps 500 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 20 \
    --load_best_model_at_end True \
    --metric_for_best_model rouge1 --greater_is_better True \
    --model_name_or_path roberta-large \
    --output_dir saves/roberta-large-samsum \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 400 --max_candidate_length 109 \
    --cache_data --disable_tqdm False \


python -m torch.distributed.launch --nproc_per_node 8  run_reranker.py --overwrite_output_dir \
    --task_name dialog --dataset_name personachat \
    --train_data_path data/personachat-large/train/gen_from_16 \
    --dev_data_path data/personachat-large/dev/gen_from_16 \
    --test_data_path data/personachat-large/test/gen_from_16 \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data True \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --warmup_ratio 0.2 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 10 \
    --load_best_model_at_end True \
    --metric_for_best_model bleu_1 --greater_is_better True \
    --model_name_or_path roberta-large \
    --output_dir saves/roberta-large-personachat \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 430 --max_candidate_length 70 --position_extend_way normal \
    --cache_data \



python -m torch.distributed.launch --nproc_per_node 8 run_reranker.py --overwrite_output_dir \
    --task_name qg --dataset_name squadqg \
    --train_data_path data/squadqg-large/train/gen_from_16 \
    --dev_data_path data/squadqg-large/dev/gen_from_16 \
    --test_data_path data/squadqg-large/test/gen_from_16 \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data True \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --warmup_ratio 0.2 \
    --evaluation_strategy steps --eval_steps 250 \
    --logging_strategy steps --logging_steps 250 \
    --save_strategy steps --save_steps 250 --save_total_limit 10 \
    --load_best_model_at_end True \
    --metric_for_best_model rougeL --greater_is_better True \
    --model_name_or_path /weizhou_data/models/roberta-large \
    --output_dir saves/roberta-large-squadqg \
    --num_cand 16 --max_num 3 \
    --loss_type contrastive \
    --max_source_length 435 --max_candidate_length 65 --position_extend_way normal \
    --cache_data \

