# ProphetNet-Dialog-Zh

This repo provides the pretrained Chinese generation model ProphetNet-Dialog-Zh.  
The details are described in [ProphetNet-X paper](https://arxiv.org/abs/2104.08006).

## Dependency
- pip install torch==1.3.0  
- pip install fairseq==v0.9.0
- pip install tensorboardX==1.7  

## Pre-trained Models

- **ProphetNet-Dialog-Zh** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_dialog_zh.pt)

Notice that this is not the model deployed for XiaoIce, similar but not the same.  
We release this model for only research purpose. You can directly use this model to generate Chinese dialog responses without finetuning.

For ProphetNet-Dialog-Zh, we use the pre-training corpus from [CDialGPT](https://github.com/thu-coai/CDial-GPT) and our internal data. Specifically, we crawled 18.2 million dyadic dialogues (conversation between two persons) longer than or equal to 2 turns(one turn denotes one utterance from one person) from the [Douban group](https://www.douban.com/group) which is a popular social networking service in China. We also load the pre-trained model from ProphetNet-Zh before pre-training, which already contains external knowledge from open-domain Chinese corpus. 
## Down-stream Tasks
We evaluate ProphetNet-Dialog-Zh on real-world XiaoIce system with human evaluation.  
We also report the resutls on  STC dataset, which you can use the [CDialGPT team cleaned version](https://github.com/thu-coai/CDial-GPT#evaluation).

## How to use

Feed "session-1 [SEP] session-2 [SEP] ... [SEP] session-n" into the encoder, predict "session-n+1" from the decoder.

The procedure includes 1) Tokenize, 2) Binarize, 3) Finetune, 4) Inference.  
ProphetNet is implemented on base of Fairseq, which you can refer to [Fairseq Mannual](https://fairseq.readthedocs.io/en/latest/command_line_tools.html).  

Tokenize. Prepare your train.src, train.tgt, and valid, test sets. Input and output of one sample are placed in the .src and .tgt file with one line.    
Use bert-uncased tokenizer to tokenize your data into word piece. 
```
import json
import tqdm
import transformers


fin_train_dev = open('finetune/STC.json', 'r', encoding='utf-8')
fout_train_src_tokenized = open('tokenized_train.src', 'w', encoding='utf-8')
fout_train_tgt_tokenized = open('tokenized_train.tgt', 'w', encoding='utf-8')

tokenizer = transformers.BertTokenizer("my_chinese_tokenizer/vocab_for_huggingface.txt")
train_dev = json.load(fin_train_dev)
train_list = train_dev['train']
for data in tqdm.tqdm(train_list):
	assert len(data) == 2
	prev = data[0].replace(" ", "").lower()
	answer = data[1].replace(" ", "").lower()
	fout_train_src.write("{}\n".format(prev))
	fout_train_tgt.write("{}\n".format(answer))
	prev_tokenized = ' '.join(tokenizer.tokenize(prev))
	answer_tokenized = ' '.join(tokenizer.tokenize(answer))
	fout_train_src_tokenized.write("{}\n".format(prev_tokenized))
	fout_train_tgt_tokenized.write("{}\n".format(answer_tokenized))
```
Binirize it with fairseq-preprocess
```
fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref tokenized_train --validpref tokenized_valid --testpref tokenized_test \
--destdir processed --srcdict prophetnet_chinese_dict/vocab_for_fairseq.txt --tgtdict prophetnet_chinese_dict/vocab_for_fairseq.txt \
--workers 20
```
Fine tune with fairseq-train.  
--disable-ngram-loss：only keep the next first token loss.  
--ngram: number of future tokens to predict. Provided pretrained checkpoint predicts 2 future tokens, and you should set it as 2 to be consistent.    
If your device does not support float16, remove --fp16.
```
DATA_DIR=processed
USER_DIR=./prophetnet
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=./model
TENSORBOARD_LOGDIR=./logs
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_dialog_zh.pt

fairseq-train \
--fp16 \
--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.00001 --min-lr 1e-09 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq 1  --max-tokens 1400 --max-sentences 7 \
--num-workers 4 \
--load-from-pretrained-model $PRETRAINED_MODEL \
--ddp-backend=no_c10d --max-epoch 10 \
--max-source-positions 512 --max-target-positions 512 \
--skip-invalid-size-inputs-valid-test \
--save-dir $SAVE_DIR \
--keep-last-epochs 10 \
--tensorboard-logdir $TENSORBOARD_LOGDIR \
$DATA_DIR
```
Inference with fairseq-generate to generate targets for given processed test files. Or you can [fairseq-interactive](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-interactive) to generate answers for your typed-in text (which should also been tokenized).
```
BEAM=5
LENPEN=1.5
CHECK_POINT=./model/checkpoint5.pt
TEMP_FILE=fairseq_outputs.txt
OUTPUT_FILE=sorted_outputs.txt

fairseq-generate processed --path $CHECK_POINT --user-dir prophetnet --task translation_prophetnet --batch-size 80 --gen-subset test --beam $BEAM --num-workers 4 --no-repeat-ngram-size 3 --lenpen $LENPEN 2>&1 > $TEMP_FILE
grep ^H $TEMP_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > $OUTPUT_FILE

```

## TIPS:
If you met problems to run fairseq-preprocess, fairseq-train and other commands, or if you want to modify the workflow/inference pipeline, 
it's a good choice to download fairseq git repo, checkout v0.9.0, and merge our codes.   
Then, modify their preprocess.py, train.py or generate.py, to run your new pipeline. 

## Repo Reference
This repo is partially referred to Fairseq-v0.9.0 and MASS.



## How to Cite
If you extend or use this work, please cite the [paper](https://arxiv.org/pdf/2001.04063) where it was introduced:
```
@inproceedings{qi2020prophetnet,
  title={Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training},
  author={Qi, Weizhen and Yan, Yu and Gong, Yeyun and Liu, Dayiheng and Duan, Nan and Chen, Jiusheng and Zhang, Ruofei and Zhou, Ming},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
  pages={2401--2410},
  year={2020}
}
@article{qi2021prophetnet,
  title={ProphetNet-X: Large-Scale Pre-training Models for English, Chinese, Multi-lingual, Dialog, and Code Generation},
  author={Qi, Weizhen and Gong, Yeyun and Yan, Yu and Xu, Can and Yao, Bolun and Zhou, Bartuer and Cheng, Biao and Jiang, Daxin and Chen, Jiusheng and Zhang, Ruofei and others},
  journal={arXiv preprint arXiv:2104.08006},
  year={2021}
}
```
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
