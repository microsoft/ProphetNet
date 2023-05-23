# Warm-up ranker

As mentioned in the paper, in order to initialize the ranker with a more general and reasonable ranking function, we increase the number of training steps and add a certain number of warm-up steps at the first ranker training iteration. Here we use the fine-tuned generator to generator the candidates for the first ranker training iteration.

## 1. preprocess data

We strongly recommend pre-tokenize the training data for the first ranker training iterantion, in order to save the training time. Taking cnndm as example, to pre-tokenize data, run:
```
EXPORT data=../data/cnndm # the derectories to save 
EXPORT dataset=cnndm 
EXPORT n=16 # number of candidate
EXPORT candidate=../warmup-generator/results/cnndm_large # the candidate path
EXPORT tokenizer_type=roberta
EXPORT tokenizer_dir=roberta-large # tokenizer path
EXPORT save=cnndm # where to save the preprocessed data

python preprocess.py --data_dir $data --dataset_name $dataset \
    --candidate_dir $candidate \
    --num_cand $n --save_name $save \
    --tokenizer_type $tokenizer_type --tokenizer_dir $tokenizer_dir
```
The above instructions will save the preprocessed data in `data/cnndm`. For the data preprocess on more datasets, check `run_prepro.sh`.

## 2. warming-up ranker

Taking cnndm as example, to warm-up ranker, run:\
```
EXPORT data=cnndm # the preprocess data path
EXPORT n=16
EXPORT model_path=roberta-large
EXPORT save_name=roberta-large-cnndm 
EXPORT source_len=400
EXPORT target_len=109

python -m torch.distributed.launch --nproc_per_node 8  run_reranker.py --overwrite_output_dir \
    --task_name sum --dataset_name $data \
    --train_data_path data/$data/train/gen_from_$n \
    --dev_data_path data/$data/train/gen_from_$n \
    --test_data_path data/$data/train/gen_from_$n \
    --do_train True --do_eval True --do_predict True --prediction_loss_only False \
    --use_untokenized_data True \ # this should be set to True
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.2 \
    --evaluation_strategy steps --eval_steps 500 \
    --logging_strategy steps --logging_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 20 \
    --load_best_model_at_end True \
    --metric_for_best_model rouge1 --greater_is_better True \
    --model_name_or_path $model_path \
    --output_dir saves/$save_name \
    --num_cand $n --max_num 3 \
    --loss_type contrastive \
    --max_source_length $source_len --max_candidate_length $target_len \
    --cache_data --disable_tqdm False \

```

The above instructions will save the warm-uped ranker in `saves/roberta-large-cnndm`. For the warming-up ranker on more datasets, check `run_train.sh`.