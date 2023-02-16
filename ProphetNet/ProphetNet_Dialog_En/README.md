# ProphetNet-Dialog-Zh

This repo provides the pretrained Chinese generation model ProphetNet-Dialog-Zh.  
The details are described in [ProphetNet-X paper](https://arxiv.org/abs/2104.08006).

## Dependency
- pip install torch==1.3.0  
- pip install fairseq==v0.9.0
- pip install tensorboardX==1.7  

## Pre-trained Models

- **ProphetNet-Dialog-En** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_dialog_en.pt)

Notice that ProphetNet-Dialog-En is pretrained with un-supervised span masked and prediction task. 
It's not an end-to-end supervised pretraining tasks, and this model should be finetuned before inference. 
The pretrained model can not be used to directly generate dialog responses as ProphetNet-Dialog-Zh. 
We plan to provide a supervised pretrained English dialog generation model in the near future.
For ProphetNet-Dialog-En, we utilize Reddit comments dataset. We firstly load the weights of ProphetNet-En then clean 60 million sessions for pre-training.

## Down-stream Tasks
For ProphetNet-Dialog-En, we carry out finetuning on [DailyDialog](https://www.aclweb.org/anthology/I17-1099.pdf) for chit-chat generation, 
[Persona-Chat](https://arxiv.org/pdf/1801.07243.pdf) for knowledge grounded conversation generation 
and [DSTC7-AVSD](https://arxiv.org/pdf/1901.09107.pdf) for conversational question answering.  
You can run get_data.sh to download these three datasets.


## How to use

The procedure includes 1) Tokenize, 2) Binarize, 3) Finetune, 4) Inference.  
ProphetNet is implemented on base of Fairseq, which you can refer to [Fairseq Mannual](https://fairseq.readthedocs.io/en/latest/command_line_tools.html).    
Hyper-parameters for each task are shown under  "finetune_scripts" folder.

## General workflow:  
Tokenize. Prepare your train.src, train.tgt, and valid, test sets. Input and output of one sample are placed in the .src and .tgt file with one line.    
Use "[SEP]" to separate different turns or to separate session and knowledge to feed input texts into the encoder, predict the response from the decoder.
```
def convert_reddit_for_finetune(fin: str, fout: str) -> None:
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    fin = open(fin, 'r', encoding='utf-8')
    # if directory not exits, make one
    make_dir(fout)
    train_src = open(os.path.join(fout, 'train.src'), 'w', encoding='utf-8')
    train_tgt = open(os.path.join(fout, 'train.tgt'), 'w', encoding='utf-8')
    valid_src = open(os.path.join(fout, 'valid.src'), 'w', encoding='utf-8')
    valid_tgt = open(os.path.join(fout, 'valid.tgt'), 'w', encoding='utf-8')
    for line in tqdm(fin, total=146832759):
        _, contexts, response = line.strip().split('\t')
        contexts = [' '.join(
            tok.tokenize(' '.join(context.strip().split()[1:]))) for context in contexts.split('EOS')]
        response = ' '.join(tok.tokenize(' '.join(response.split(' ')[1:])))
        if np.random.rand() < 0.05:
            valid_src.write(' [SEP] '.join(contexts) + '\n')
            valid_tgt.write(response + '\n')
        else:
            train_src.write(' [SEP] '.join(contexts) + '\n')
            train_tgt.write(response + '\n')

```
Binirize it with fairseq-preprocess
```
fairseq-preprocess \
--user-dir ./prophetnet \
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
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_dialog_en.pt

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
