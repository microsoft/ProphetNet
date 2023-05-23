# Data acquisition for JGR

## 1. CNN/Dailymail

We use the non-anonymized version of CNN/Dailymail. First down load the [url_lists](https://github.com/abisee/cnn-dailymail) of train/dev/test set, and down load the unzip the stories directories from [here](https://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Then put the uzipped directories to `/data/cnndm/raw_data`.

Then run the following instructions:
```
cd cnndm
python generate_data.py
```
This instrcutions will finally generate `train/dev/test_data.json`, which contain the training/dev/test samples of CNN/Dailymail.

## 2. SAMSum

First download and unzip the data files of SAMSum from [here](https://arxiv.org/src/1911.12237v2/anc/corpus.7z), then put them to `/data/samsam/raw_data`

run:
```
cd samsum
python generate_data.py
```
This instrcutions will finally generate `train/dev/test_data.json`, which contain the training/dev/test samples of SAMSum.

## 3. Squadqg & Personachat

We use the preprocessed version of squadqg and personachat from [GLGE](https://github.com/microsoft/glge#get-dataset). You should first download the training/dev set of squadqg/personachat from [here](https://drive.google.com/file/d/1F4zppa9Gqrh6iNyVsZJkxfbm5waalqEA/view) and test set from [here](https://drive.google.com/file/d/11lDXIG87dChIfukq3x2Wx4r5_duCRm_J/view). The put the `org_data` directory to the corresponding folder. Then run:

```
python prepro.py --dataset_name squadqg # squadqg
python prepro.py --dataset_name personachat # personachat
```
This instrcutions will finally generate `train/dev/test_data.json`, which contain the training/dev/test samples of Squadqg/Personachat.

