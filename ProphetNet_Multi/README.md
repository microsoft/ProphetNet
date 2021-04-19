# ProphetNet-Multi

This repo provides the pretrained multi-lingual generation model ProphetNet-multi.  
The details are described in [ProphetNet-X paper](https://arxiv.org/abs/2104.08006).

## Dependency

- pip install torch==1.3.0  
- pip install fairseq==v0.9.0  
- pip install tensorboardX==1.7  
- pip install sentencepiece  
- pip install sacrebleu

## Pre-trained Models

Recommended Checkpoints:
- **ProphetNet-Multi** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_multi.pt)

Expired Checkpoints:
- **ProphetNet-Multi-Wiki100(Baseline Model Unicoder-FNP for XGLUE)** [[link]](https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_multi_wiki100.pt)

For ProphetNet-Multi, we select 52 common languages to collect and clean multi-lingual data from Wiki-100 and Common Crawl. After cleaning and tokenizing, the pre-training corpus size is 1.5TB. The detailed description can be found in ProphetNet-X(To appear soon).  
For ProphetNet-Multi-Wiki100(Baseline Model Unicoder-FNP for XGLUE), it is pretrained with 100 languages Wikipedia data Wiki-100 described in [XGLUE](https://arxiv.org/abs/2004.01401).


## Down-stream Tasks
Cross-lingual generation task [XGLUE](https://arxiv.org/abs/2004.01401), details about XGLUE can be found in [XGLUE github repo](https://github.com/microsoft/XGLUE) and the dataset can be downloaded from [link](https://microsoft.github.io/XGLUE/).  

To use XProphetNet:  
1) Preprocess. Tokenize the text with provided bpe model and generate the binary file for Fairseq.  
2) Finetune. Finetune the model with the supervised corpus, eg English News Title Generation or English Question Generation.
3) Inference. Use finetuned checkpoint with less epoches to generate outputs for zero-shot languages, and checkpoint with more epoches for supervised language.


## Preprocess
Download the [XGLUE dataset](https://microsoft.github.io/XGLUE/), uncompress, and put cross-lingual generation NTG and QG folders under ./finetune_data.  
[xprophetnet_preprocess.sh](https://github.com/microsoft/ProphetNet/blob/master/ProphetNet_Multi/xprophetnet_preprocess.sh) will 1) tokenize with the given bpe model. 2) generate the desired binary files.  
It should be noticed that although XProphetNet support maximum 512 input and output tokens, first 256 tokens are kept to be consistent with other baseline models.  
After preprocess, three types of files will appear under ./finetune_data folder:  
1) NTG folder and QG folder which contain the original text. 
2) NTG_tokenized and QG_tokenized which contain the tokenized text.  
3) NTG_processed_* and QG_processed_* which contain the binary files for Fairseq to finetune and inference.

## Finetune
With the preprocessed files under ./finetune_data and the pretrained checkpoint under ./models/ , use [finetune_xprophetnet_qg.sh](https://github.com/microsoft/ProphetNet/blob/master/ProphetNet_Multi/finetune_xprophetnet_qg.sh) and [finetune_xprophetnet_ntg.sh](https://github.com/microsoft/ProphetNet/blob/master/ProphetNet_Multi/finetune_xprophetnet_ntg.sh) to finetune ProphetNet_Multi.  
It should be noticed that the finetuning batch size is 1024. 
--update-freq means accumulate this steps to update the parameters. 
--max-sentences means maximum samples per GPU. 
Thus the batch size will be the product of update_freq, max_sentences and the number of your GPU.  
The model is finetuned on four 24GB P40 GPU.

## Inference
Inference your model with [inference_qg.sh](https://github.com/microsoft/ProphetNet/blob/master/ProphetNet_Multi/inference_qg.sh) and [inference_ntg.sh](https://github.com/microsoft/ProphetNet/blob/master/ProphetNet_Multi/finetune_xprophetnet_ntg.sh) which generate the outputs and evaluate with the original text.  
Four types of files will be generated under ./outputs folder.  
1) output_NTG_xxxxx.txt: The original outputs from Fairseq-generate.  
2) sort_hypo_NTG_xxxxxx.txt: The sorted hypothesis generated from XProphetNet, which is extracted from the former file.  
3) sort_hypo_NTG_xxxxxx.txt.post: The post-processed hypothesis file, which is recovered from the former file with bpe model
4) score_NTG_xxxxx.txt: The BLEU score  calculated with the former file and the original raw text.

It should be noticed that to generate outputs for given input, less epoches finetuned checkpoint will be used for zero-shot language,
and more epoches finetuned checkpoint will be used for the supervised language input.  
For example, xprophetnet_qg_en is finetuned on English Question Gneration data, 
checkpoint17.pt is used to generate outputs for the supervised language english input, 
while checkpoint6.pt is used to generate outputs for other zero-shot language input.


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
```
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
