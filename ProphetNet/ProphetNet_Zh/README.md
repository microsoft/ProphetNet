# ProphetNet-Zh

This repo provides the pretrained Chinese generation model ProphetNet-Zh.  
The details are described in [ProphetNet-X paper](https://arxiv.org/abs/2104.08006).

## Dependency
- pip install torch==1.3.0  
- pip install fairseq==v0.9.0  
- pip install tensorboardX==1.7    

## Pre-trained Models

- **ProphetNet-Zh** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_zh.pt)

 For ProphetNet-Zh, we collect Chinese Wikipedia, [CLUE](https://github.com/CLUEbenchmark/CLUE) and Chinese Common Crawl data to reach 160GB. For traditional Chinese data, we firstly use [OpenCC](https://github.com/BYVoid/OpenCC) to convert them to simplified Chinese. The pre-training corpus includes common webs, online forums, comments websites, Q\&A websites, Chinese Wikipedia, and other encyclopedia websites. We build a simplified Chinese char dictionary. The char vocabulary size is 9,360.

## Down-stream Tasks
For now, ProphetNet-Zh is finetuned with quetion answering task [MATINF-QA](https://arxiv.org/abs/2004.12302), 
summarization tasks [MATINF-SUMM](https://arxiv.org/abs/2004.12302) and [LCSTS](https://arxiv.org/abs/1506.05865).  
To get the raw data for MATINF-QA and MATINF-SUMM, you can visit [this website](https://github.com/WHUIR/MATINF), fill the agreement form then get the download link.  
To get the raw data for LCSTS, you can visit [this website](http://icrc.hitsz.edu.cn/Article/show/139.html), fill the agreement form, email the dataset authors then get the download link.  

## How to use

The procedure includes 1) Tokenize, 2) Binarize, 3) Finetune, 4) Inference.  
ProphetNet is implemented on base of Fairseq, which you can refer to [Fairseq Mannual](https://fairseq.readthedocs.io/en/latest/command_line_tools.html).  

Tokenize. Prepare your train.src, train.tgt, and valid, test sets. Input and output of one sample are placed in the .src and .tgt file with one line.    

```
import transformers import BertTokenizer

def prophetnet_zh_tokenize(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer("prophetnet_chinese_dict/vocab_for_huggingface.txt")
    for line in fin:
        word_pieces = tok.tokenize(line.strip())
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))

prophetnet_zh_tokenize('train.src', 'tokenized_train.src')
prophetnet_zh_tokenize('train.tgt', 'tokenized_train.tgt')
prophetnet_zh_tokenize('valid.src', 'tokenized_valid.src')
prophetnet_zh_tokenize('valid.tgt', 'tokenized_valid.tgt')
prophetnet_zh_tokenize('test.src', 'tokenized_test.src')
prophetnet_zh_tokenize('test.tgt', 'tokenized_test.tgt')
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
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_zh.pt

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
