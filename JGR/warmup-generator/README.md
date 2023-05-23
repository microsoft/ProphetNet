# warm up generator

To achieve a better performance of JGR, it's neccessary to pre-finetune the generator with MLE loss on the target training set. This folder contains the code for warming-up bart on the target dataset.

## 1. Fine-tune bart
Taking cnndm as example, to fine-tune bart on cnndm, run:
```
EXPORT model_path=bart-large # the pre-trained model path

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_train.py \
--dataset_name cnndm \
--report_to none \
--do_train True --do_eval True --do_predict True \
--evaluation_strategy epoch  --logging_strategy epoch --save_strategy epoch \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--num_train_epochs 5 \
--metric_for_best_model rouge2 \
--generation_num_beams 4 --generation_max_length 100 \
--model_name_or_path $model_path \
--config_name $model_path \
--tokenizer_name $model_path \
--max_source_length 1020 --max_target_length 100 \
--output_dir saves/bart-large-cnndm  --overwrite_output_dir
```
The above code will store the fine-tuned bart checkpoint in `saves/bart-large-cnndm`. For fine-tuning bart on more datasets, please check `run_train.sh`.

## 2. Generate candidates for warming-up ranker
As mentioned in the paper, in order to initialize the ranker with a more general and reasonable ranking function, we increase the number of training steps and add a certain number of warm-up steps at the first ranker training iteration. Here we use the fine-tuned generator to generator the candidates for the first ranker training iteration.
Taking cnndm as example:
```
EXPORT model_path=saves/bart-large-cnndm # The fine-tuned bart
EXPORT save_name=cnndm-large # Save name of the dataset
EXPORT num_cand=16 # num of candidate generated

# for training set use sampling
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name cnndm \
--split train \
--use_tokenized_data True \
--do_predict \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_do_sample True \
--generation_num_beams 1 \
--generation_num_return_sequences $num_cand \
--model_name_or_path $model_path \
--config_name $model_path \
--tokenizer_name $model_path \
--max_source_length 1020 --max_target_length 100 --disable_tqdm False \
--output_dir predict --save_name $save_name

# for dev and test, use beam search
python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name cnndm \
--split dev \
--use_tokenized_data True \
--do_predict \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_do_sample False \
--generation_num_beams $num_cand \
--generation_num_return_sequences $num_cand \
--model_name_or_path $model_path \
--config_name $model_path \
--tokenizer_name $model_path \
--max_source_length 1020 --max_target_length 100 --disable_tqdm False \
--output_dir predict --save_name $save_name

python -m torch.distributed.launch --nproc_per_node 8 --master_port 6006 run_diverse_gen.py  --overwrite_output_dir \
--dataset_name cnndm \
--split test \
--use_tokenized_data True \
--do_predict \
--dataloader_pin_memory True --predict_with_generate True --generation_max_length 100 \
--generation_do_sample False \
--generation_num_beams $num_cand \
--generation_num_return_sequences $num_cand \
--model_name_or_path $model_path \
--config_name $model_path \
--tokenizer_name $model_path \
--max_source_length 1020 --max_target_length 100 --disable_tqdm False \
--output_dir predict --save_name $save_name
```

The above instructions will generate candidates, and store them in `results/$save_name`. For generating candidates on more datasets, please check `run_generate.sh`.