# ProphetNet-X

1. This repo provides the code for reproducing the experiments in [*ProphetNet*](https://arxiv.org/pdf/2001.04063). In the paper, we propose a new pre-trained language model called ProphetNet for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction. 

2. We have released the ProphetNet baselines for [GLGE](https://github.com/microsoft/glge) benchmark ([A New General Language Generation Evaluation Benchmark](https://arxiv.org/abs/2011.11928)) in [here](./GLGE_baselines). Have a try! :) 

3. We provide [ProphetNet-X family models](https://arxiv.org/abs/2104.08006) for Chinses(ProphetNet-Zh), Multi-lingual(ProphetNet-Multi), English open domain dialog(ProphetNet-Dialog), Chinese open domain dialog(ProphetNet-Dialog-Zh), code generation(ProphetNet-Code). The details are described in [ProphetNet-X paper](https://arxiv.org/abs/2104.08006). Different ProphetNet-X models have the only difference of the vocabulary file. Simply modify one model file and you can evaluate your idea with all the pretrained models and finetuning scripts!

This repo is still developing, feel free to report bugs and we will fix them ~

## News

- **EL-Attention: Memory Efficient Lossless Attention for Generation**, Yu Yan, Jiusheng Chen, Weizhen Qi, Nikhil Bhendawade, Yeyun Gong, Nan Duan, Ruofei Zhang, ***ICML 2021***, [Code](https://github.com/microsoft/fastseq/blob/main/examples/EL-attention/README.md) [Paper](https://arxiv.org/abs/2105.04779)  
- **BANG: Bridging Autoregressive and Non-autoregressive Generation with Large Scale Pretraining**, Weizhen Qi, Yeyun Gong, Jian Jiao, Yu Yan, Weizhu Chen, Dayiheng Liu, Kewen Tang, Houqiang Li, Jiusheng Chen, Ruofei Zhang, Ming Zhou, Nan Duan, ***ICML 2021***, [Code](https://github.com/microsoft/BANG) [Paper](https://arxiv.org/abs/2012.15525)  
- **GLGE: A New General Language Generation Evaluation Benchmark**, Dayiheng Liu, Yu Yan, Yeyun Gong, Weizhen Qi, Hang Zhang, Jian Jiao, Weizhu Chen, Jie Fu, Linjun Shou, Ming Gong, Pengcheng Wang, Jiusheng Chen, Daxin Jiang, Jiancheng Lv, Ruofei Zhang, Winnie Wu, Ming Zhou, Nan Duan, ***ACL 2021 Findings***, [Code](https://github.com/microsoft/glge) [Paper](https://arxiv.org/abs/2011.11928) [Leaderboard](https://microsoft.github.io/glge/)  
- **Mask Attention Networks: Rethinking and Strengthen Transformer**, Zhihao Fan, Yeyun Gong, Dayiheng Liu, Zhongyu Wei, Siyuan Wang, Jian Jiao, Nan Duan, Ruofei Zhang, Xuanjing Huang, ***NAACL 2021***,  [Code](https://github.com/LibertFan/MAN) [Paper](https://arxiv.org/abs/2103.13597v1)  
- **An Enhanced Knowledge Injection Model for Commonsense Generation**, Zhihao Fan, Yeyun Gong, Zhongyu Wei, Siyuan Wang, Yameng Huang, Jian Jiao, Xuanjing Huang, Nan Duan, Ruofei Zhang, ***COLING 2020***, [Code](https://github.com/LibertFan/EKI-BART) [Paper](https://arxiv.org/abs/2012.00366)  
- **Tell Me How to Ask Again: Question Data Augmentation with Controllable Rewriting in Continuous Space**, Dayiheng Liu, Yeyun Gong, Jie Fu, Yu Yan, Jiusheng Chen, Jiancheng Lv, Nan Duan, Ming Zhou, ***EMNLP 2020***,  [Code](https://github.com/dayihengliu/CRQDA) [Paper](https://aclanthology.org/2020.emnlp-main.467.pdf)  
- **Diverse, Controllable, and Keyphrase-Aware: A Corpus and Method for News Multi-Headline Generation**, Dayiheng Liu, Yeyun Gong, Jie Fu, Wei Liu, Yu Yan, Bo Shao, Daxin Jiang, Jiancheng Lv, Nan Duan, ***EMNLP 2020***, [Code](https://github.com/dayihengliu/KeyMultiHeadline) [Paper](https://aclanthology.org/2020.emnlp-main.505.pdf)  




## Dependency
- pip install torch==1.3.0  
- pip install fairseq==v0.9.0
- pip install tensorboardX==1.7  

## Pre-trained Models

We have released the following checkpoints for pre-trained models as described in the paper of ProphetNet-X(appear soon).

ProphetNet-X is based on [ProphetNet](https://arxiv.org/pdf/2001.04063), which also serves the ProphetNet-En model.


Recommended Checkpoints:
- **ProphetNet-En** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_en.pt)
- **ProphetNet-Zh** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_zh.pt)
- **ProphetNet-Multi** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_multi.pt)
- **ProphetNet-Dialog-En** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_dialog_en.pt)
- **ProphetNet-Dialog-Zh** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_dialog_zh.pt)
- **ProphetNet-Code** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_code.pt)

Expired Checkpoints:
- **ProphetNet-En-16GB** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_en_16g.pt)
- **ProphetNet-Multi-Wiki100** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_multi_wiki100.pt)

## How to use

The procedure includes 1) Tokenize, 2) Binarize, 3) Finetune, 4) Inference.  
ProphetNet is implemented on base of Fairseq, which you can refer to [Fairseq Mannual](https://fairseq.readthedocs.io/en/latest/command_line_tools.html).  

**For all the ProphetNet-X models, the only difference is the dictionary, which means different Tokenizers should be used.**

We take ProphetNet-En for example:

Tokenize. Prepare your train.src, train.tgt, and valid, test sets. Input and output of one sample are placed in the .src and .tgt file with one line.    
Use bert-uncased tokenizer to tokenize your data into word piece. 
```
from transformers import BertTokenizer


def bert_uncased_tokenize(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in fin:
        word_pieces = tok.tokenize(line.strip())
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))
bert_uncased_tokenize('train.src', 'tokenized_train.src')
bert_uncased_tokenize('train.tgt', 'tokenized_train.tgt')
bert_uncased_tokenize('valid.src', 'tokenized_valid.src')
bert_uncased_tokenize('valid.tgt', 'tokenized_valid.tgt')
bert_uncased_tokenize('test.src', 'tokenized_test.src')
bert_uncased_tokenize('test.tgt', 'tokenized_test.tgt')
```
Binirize it with fairseq-preprocess
```
fairseq-preprocess \
--user-dir prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref tokenized_train --validpref tokenized_valid --testpref tokenized_test \
--destdir processed --srcdict vocab.txt --tgtdict vocab.txt \
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
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_en.pt

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
