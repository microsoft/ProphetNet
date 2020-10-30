# ProphetNet

This repo provides the code for reproducing the experiments in [*ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training*](https://arxiv.org/pdf/2001.04063). In the paper, we propose a new pre-trained language model called ProphetNet for sequence-to-sequence learning with a novel self-supervised objective called future n-gram prediction. 


## Dependency
- pip install torch==1.3.0  
- pip install fairseq==v0.9.0

## Pre-trained Models

We have released the following checkpoints for pre-trained models as described in the [paper](https://arxiv.org/pdf/2001.04063): 

- **ProphetNet-large-16GB** (pre-trained on 16GB corpus Wikipedia + BookCorpus with 64 epochs) [[link]](https://drive.google.com/file/d/1PctDAca8517_weYUUBW96OjIPdolbQkd/view?usp=sharing)
- **ProphetNet-large-160GB** (pre-trained on 160GB corpus Wikipedia + BookCorpus + RealNews + OpenWebText + Stories with 14 epochs) [[link]](https://drive.google.com/file/d/1_nZcF-bBCQvBBcoPzA1nPZsz-Wo7hzEL/view?usp=sharing)
   
 
For future work, we will employ ProphetNet into other pre-trained language model settings, try more future tokens, increase the corpus size, and adjust the weight of different future tokens.

## Experiments
We carried out experiments on [CNN / Daily Mail](https://github.com/harvardnlp/sent-summary), [Gigaword](https://github.com/harvardnlp/sent-summary), and Question generation - [SQuAD](https://arxiv.org/abs/1806.03822).  
 
## [CNN / Daily Mail](https://github.com/harvardnlp/sent-summary)

### Data Preprocess
We use the pre-processed data of [UniLM](https://github.com/microsoft/unilm),
 which can be downloaded from [the provided link of UniLM](https://drive.google.com/open?id=1jiDbDbAsqy_5BM79SmX6aSu5DQVCAZq1).     
 According to the scripts from [UniLM issue](https://github.com/microsoft/unilm/issues/11), we use preprocess_cnn_dm.py to tokenize CNN/DailyMail data.  
 After this, we generate the binary data files with this script:
 ```
fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref cnndm/prophetnet_tokenized/train --validpref cnndm/prophetnet_tokenized/valid --testpref cnndm/prophetnet_tokenized/test \
--destdir cnndm/processed --srcdict ./vocab.txt --tgtdict ./vocab.txt \
--workers 20
```



### Fine-tune
We fine-tuned the model on 8 * NVIDIA V100 (16GB) GPUs. Note that batch size = 8 (GPUS) * 2 (sample per GPU) * 32 (accumulate) = 512. To finetune the model on CNN/Daily Mail, please run:
```
DATA_DIR=cnndm/processed
USER_DIR=./prophetnet
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=cnndm/finetune_cnndm_checkpoints
TENSORBOARD_LOGDIR=cnndm/finetune_cnndm_tensorboard
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt

fairseq-train \
--fp16 \
--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.0001 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq 32  --max-sentences 2 \
--num-workers 4 \
--load-from-pretrained-model $PRETRAINED_MODEL \
--load-sep \
--ddp-backend=no_c10d --max-epoch 10 \
--max-source-positions 512 --max-target-positions 512 \
--skip-invalid-size-inputs-valid-test \
--seed 1 \
--save-dir $SAVE_DIR \
--keep-last-epochs 10 \
--tensorboard-logdir $TENSORBOARD_LOGDIR \
$DATA_DIR
```


### Inference and Evaluation
After fine-tuning, the scripts of inference and evaluation are as follows:
```
SUFFIX=_ck9_pelt1.2_test_beam5
BEAM=5
LENPEN=1.2
CHECK_POINT=cnndm/finetune_cnndm_checkpoints/checkpoint9.pt
OUTPUT_FILE=cnndm/output$SUFFIX.txt
SCORE_FILE=cnndm/score$SUFFIX.txt

fairseq-generate cnndm/processed --path $CHECK_POINT --user-dir prophetnet --task translation_prophetnet --batch-size 32 --gen-subset test --beam $BEAM --num-workers 4 --min-len 45 --max-len-b 110 --no-repeat-ngram-size 3 --lenpen $LENPEN 2>&1 > $OUTPUT_FILE

grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > cnndm/sort_hypo$SUFFIX.txt
python cnndm/eval/postprocess_cnn_dm.py --generated cnndm/sort_hypo$SUFFIX.txt --golden cnndm/original_data/test.summary > $SCORE_FILE
```

The results on CNN/DailyMail test set are shown in this Table:

| Model                            | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- | --------- |
| [MASS(16G)](https://arxiv.org/pdf/1905.02450.pdf)                      | 42.12     | 19.50     | 39.01     |
| [UniLM(16G)](https://arxiv.org/pdf/1905.03197.pdf)                       | 43.33     | 20.21     | 40.51     |
| **ProphetNet(16G)**              | 43.68     | 20.64      | 40.72     |
| [T5(750G)](https://arxiv.org/pdf/1910.10683.pdf)                         | 43.52     | **21.55** | 40.69     |
| [PEGASUSLARGE (C4, 750G)](https://arxiv.org/pdf/1912.08777.pdf)          | 43.90     | 21.20     | 40.76     |
| [PEGASUSLARGE (HugeNews, 3800G)](https://arxiv.org/pdf/1912.08777.pdf)   | 44.17     | 21.47     | 41.11     |
| [BART(160G)](https://arxiv.org/pdf/1910.13461.pdf)                       | 44.16     | 21.28     | 40.90 |
| **ProphetNet(160G)**             | **44.20** | 21.17     | **41.30**     |

### Checkpoint
- ProphetNet-large-160GB (fine-tuned on CNN/Daily Mail with 9 epochs) [[link]](https://drive.google.com/file/d/14v0HMc7obh_5aPFSFWzcr_nZCrK49Sey/view?usp=sharing)



## [Gigaword](https://github.com/harvardnlp/sent-summary)

### Data Preprocess
We use the pre-processed data of [UniLM](https://github.com/microsoft/unilm), which can be downloaded from [the provided link of UniLM](https://drive.google.com/open?id=1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV).  
Put the downloaded UniLM BERT-cased tokenized files into gigaword/unilm_tokenized, put the downloaded [original un-tokenized files](https://github.com/harvardnlp/sent-summary) (which are also provided in [UniLM processed data](https://drive.google.com/open?id=1USoQ8lJgN8kAWnUnRrupMGrPMLlDVqlV) ) into gigaword/original_data for evaluation.  
Use the following script to generate BERT-uncased files.

```
from pytorch_transformers import BertTokenizer
import tqdm

def convert_cased2uncased(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in tqdm.tqdm(fin.readlines()):
        org = line.strip().replace(" ##", "")
        new = tok.tokenize(org)
        new_line = " ".join(new)
        fout.write('{}\n'.format(new_line))
convert_cased2uncased('gigaword/unilm_tokenized/train.src', 'gigaword/prophetnet_tokenized/train.src')
...
```
then generate the binary trainable files, 
```
fairseq-preprocess \
--user-dir prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref gigaword/prophetnet_tokenized/train --validpref gigaword/prophetnet_tokenized/dev --testpref gigaword/prophetnet_tokenized/test \
--destdir gigaword/processed --srcdict vocab.txt --tgtdict vocab.txt \
--workers 20
```

### Fine-tune
We fine-tuned the model on 8 * NVIDIA V100 (16GB) GPUs.  To fine-tune the model on Gigaword, please run:
```
DATA_DIR=gigaword/processed
USER_DIR=./prophetnet
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=gigaword/finetune_gigaword_checkpoints
TENSORBOARD_LOGDIR=gigaword/finetune_gigaword_tensorboard
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt

fairseq-train $DATA_DIR \
	--fp16 \
	--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
	--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
	--lr 0.0001 --min-lr 1e-09 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
	--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	--criterion $CRITERION --label-smoothing 0.1 \
	--update-freq 1  --max-tokens 1300 --max-sentences 16 \
	--num-workers 8  \
	--load-from-pretrained-model $PRETRAINED_MODEL \
	--ddp-backend=no_c10d --max-epoch 10 \
	--max-source-positions 512 --max-target-positions 512 \
	--skip-invalid-size-inputs-valid-test \
	--save-dir $SAVE_DIR \
	--keep-last-epochs 10 \
	--tensorboard-logdir $TENSORBOARD_LOGDIR \
```

 

### Inference and Evaluation
Download the evaluation scripts from [provided evaluation scripts of UniLM](https://github.com/microsoft/unilm/tree/master/unilm-v1/src/gigaword) and put them in gigaword/eval.
The scripts of inference and evaluation are as follows:

```
SUFFIX=_ck7_pelt1.0_test_beam4
BEAM=4
LENPEN=1.0
CHECK_POINT=gigaword/finetune_gigaword_checkpoints/checkpoint7.pt
OUTPUT_FILE=gigaword/output$SUFFIX.txt
SCORE_FILE=gigaword/score$SUFFIX.txt

fairseq-generate gigaword/processed --path $CHECK_POINT --user-dir prophetnet --task translation_prophetnet --batch-size 80 --gen-subset test --beam $BEAM --num-workers 4 --lenpen $LENPEN 2>&1 > $OUTPUT_FILE
grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > gigaword/sort_hypo$SUFFIX.txt
python gigaword/eval/eval.py --pred gigaword/sort_hypo$SUFFIX.txt --gold gigaword/original_data/test.tgt.txt --perl > $SCORE_FILE
```

For gigaword datase, the test set is small which is sensitive to hyper-parameters.  Different models get their best performances with different inference length penalty and beam size.  
Results on dev set are more stable, which concains about 100 times more pieces of data than test set. We set the hyper-parameters according to performance on dev set.  
The results on Gigaword test set are shown in this Table:

| Model    | ROUGE-1   | ROUGE-2   | ROUGE-L   |
| ------------------------------------------------------------------- | --------- | --------- | --------- |
| [MASS(16G)](https://arxiv.org/pdf/1905.02450.pdf) | 38.73     | 19.71      | 35.96     |
| [UniLM(16G)](https://arxiv.org/pdf/1905.03197.pdf) | 38.45     | 19.45      | 35.75     |
| [PEGASUSLARGE (C4,750G)](https://arxiv.org/pdf/1912.08777.pdf) | 38.75     | 19.96      | 36.14    |
| [PEGASUSLARGE (HugeNews,3800G)](https://arxiv.org/pdf/1912.08777.pdf) | 39.12    | 19.86      | 36.24     |
| ProphetNet(16G) | **39.55**     | 20.27      | 36.57     |
| **ProphetNet(160G)** | 39.51 | **20.42** | **36.69** |

If we slightly adjust the hyper-parameters into beam size 5, the results for ProphetNet(160G) will be 39.62(R1), 20.58(R2), 36.77(RL).    



### Checkpoint
- ProphetNet-large-160GB (fine-tuned on Gigawords with 7 epochs) [[link]](https://drive.google.com/file/d/1NFVOn89pML6--0MvIb-TzhFFvtdEA0gc/view?usp=sharing)


## Question Generation - [SQuAD](https://arxiv.org/abs/1806.03822)

### Data Preprocess
We use the pre-processed data of [UniLM](https://github.com/microsoft/unilm), which can be downloaded from [the provided link of UniLM](https://drive.google.com/open?id=11E3Ij-ctbRUTIQjueresZpoVzLMPlVUZ).   
Put the UniLM tokenized files into qg/unilm_tokenized, and [original files](https://github.com/xinyadu/nqg/tree/master/data/processed) into qg/original_data for evaluation.  
Slightly different from UniLM of using "paragraph [SEP] answer" as input to generate question, we use "answer [SEP] paragraph" as our input sequence, because the first 512 tokens are fed to our model, and we don't want to drop the answer span.

```
def convert_cased2uncased_reverse(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in tqdm.tqdm(fin.readlines()):
        org = line.strip().replace(" ##", "").split("[SEP]")
        ans = tok.tokenize(org[1].strip())
        par = tok.tokenize(org[0].strip())[:510 - len(ans)] # at most 512 tokens can be fed to our model
        par = " ".join(par)
        ans = " ".join(ans)
        fout.write('{} [SEP] {}\n'.format(ans, par))
convert_cased2uncased('qg/unilm_tokenized/train.q.tok.txt', 'qg/prophetnet_tokenized/train.tgt')
convert_cased2uncased_reverse('qg/unilm_tokenized/train.pa.tok.txt', 'qg/prophetnet_tokenized/train.src')
```

Then, use fairseq-preprocess to generate the binary data files into qg/processed.

### Fine-tune
We fine-tuned the model with 4 * NVIDIA V100 (16GB) GPUs, on each GPU the max batch size is set to 7 to avoid OOM. Thus the total batch size is 28.  
To fine-tune the ProphetNet, please run:
```
DATA_DIR=qg/processed
USER_DIR=./prophetnet
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=qg/finetune_qg_checkpoints
TENSORBOARD_LOGDIR=qg/finetune_qg_tensorboard
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt

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

### Inference and Evaluation
Download the evaluation scripts from [provided evaluation scripts of UniLM](https://github.com/microsoft/unilm/tree/master/src/qg) and put them in qg/eval.  
Notice that the evaluation files are not complete, the rest can be downloaded from [original eval files](https://github.com/xinyadu/nqg/tree/master/qgevalcap) which are coded in python2.
The scrpits of inference and evaluation are as follows:
```
SUFFIX=_ck5_pelt1.5_test_beam5
BEAM=5
LENPEN=1.5
CHECK_POINT=qg/finetune_qg_checkpoints/checkpoint5.pt
OUTPUT_FILE=qg/output$SUFFIX.txt
SCORE_FILE=qg/score$SUFFIX.txt

fairseq-generate qg/processed --path $CHECK_POINT --user-dir prophetnet --task translation_prophetnet --batch-size 80 --gen-subset test --beam $BEAM --num-workers 4 --no-repeat-ngram-size 3 --lenpen $LENPEN 2>&1 > $OUTPUT_FILE
grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > qg/sort_hypo$SUFFIX.txt

# switch into python 2
python qg/eval/eval_on_unilm_tokenized_ref.py --out qg/sort_hypo$SUFFIX.txt --src qg/prophetnet_tokenized/test.src --tgt qg/prophetnet_tokenized/test.tgt  > $SCORE_FILE
python qg/eval/eval.py --out qg/sort_hypo$SUFFIX.txt --src qg/prophetnet_tokenized/test.src --tgt qg/original_data/tgt-test.txt  >> $SCORE_FILE
```


We report the results with the [original tokenized references](https://github.com/xinyadu/nqg/tree/master/data/processed).  
The same model and hyper-parameters are used to generate the results for two different data splits.  
Results of [data split](https://github.com/xinyadu/nqg/tree/master/data):

| Model                                                              | BLEU-4    | METEOR    | ROUGE-L   |
| ------------------------------------------------------------------ | --------- | --------- | --------- |
| [CorefNQG](https://www.aclweb.org/anthology/P18-1177) | 15.16     | 19.12     | -         |
| [SemQG](https://arxiv.org/pdf/1909.06356.pdf)   | 18.37     | 22.65     | 46.68     |
| [UniLM(16G)](https://arxiv.org/abs/1905.03197)                          | 21.63 | 25.04 | 51.09 |
| **ProphetNet(16G)**                                                     | **23.91** | **26.60** | **52.26** |

Results of [another data split](https://aclweb.org/anthology/D18-1424), which uses reversed dev set and test set:

| Model                                                            | BLEU-4    | METEOR    | ROUGE-L   |
| ---------------------------------------------------------------- | --------- | --------- | --------- |
| [MP-GSN](https://aclweb.org/anthology/D18-1424)     | 16.38     | 20.25     | 44.48     |
| [SemQG](https://arxiv.org/pdf/1909.06356.pdf) | 20.76     | 24.20     | 48.91     |
| [UniLM(16G)](https://arxiv.org/abs/1905.03197)                        | 23.08 | 25.57 | 52.03 |
| **ProphetNet(16G)**                                                   | **25.80** | **27.54** | **53.69** |

### Checkpoint

- ProphetNet-large-16GB (fine-tuned on SQuAD with 5 epochs) [[link]](https://drive.google.com/file/d/1IiutfQp_Q5ggQErcdKd2byuAEnwzC09I/view?usp=sharing) 

## Fine tune on other datasets

Similarly, the procedure includes 1) Tokenize, 2) Binarize, 3) Finetune, 4) Inference.  
ProphetNet is implemented on base of Fairseq, which you can refer to [Fairseq Mannual](https://fairseq.readthedocs.io/en/latest/command_line_tools.html).  
Prepare your train.src, train.tgt, and valid, test sets. Input and output of one sample are placed in the .src and .tgt file with one line.    
Use bert-uncased tokenizer to tokenize your data into word piece.
```
from pytorch_transformers import BertTokenizer

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
--load-sep: load pretrained seperation token into [X_SEP]. (Each sample take one line, you can use [X_SEP] to seperate sentences in one line. CNN/DM finetuning used it.)  
--ngram: number of future tokens to predict. Provided pretrained checkpoint predicts 2 future tokens, and you should set it as 2 to be consistent.  
```
DATA_DIR=processed
USER_DIR=./prophetnet
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=./model
TENSORBOARD_LOGDIR=./logs
PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt

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

fairseq-generate qg/processed --path $CHECK_POINT --user-dir prophetnet --task translation_prophetnet --batch-size 80 --gen-subset test --beam $BEAM --num-workers 4 --no-repeat-ngram-size 3 --lenpen $LENPEN 2>&1 > $TEMP_FILE
grep ^H $TEMP_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > $OUTPUT_FILE

```
## Repo Reference
This repo is partially referred to Fairseq-v0.9.0 and MASS.

## How to Cite
If you extend or use this work, please cite the [paper](https://arxiv.org/pdf/2001.04063) where it was introduced:
```
@article{yan2020prophetnet,
  title={ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training},
  author={Yan, Yu and Qi, Weizhen and Gong, Yeyun and Liu, Dayiheng and Duan, Nan and Chen, Jiusheng and Zhang, Ruofei and Zhou, Ming},
  journal={arXiv preprint arXiv:2001.04063},
  year={2020}
}
```
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
