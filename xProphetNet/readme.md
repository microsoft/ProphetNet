# XProphetNet
This is the implement of FNP(Future N-grams Prediction) cross-lingual generation model for [XGLUE](https://arxiv.org/abs/2004.01401) baseline. More details about XGLUE can be found in [XGLUE github repo](https://github.com/microsoft/XGLUE) and the dataset can be downloaded from [link](https://microsoft.github.io/XGLUE/).  
This is also the cross-lingual version ProphetNet.  
We pretrained ProphetNet on 100 languages Wikipedia corpus with 350k steps, which can be downloaded from [link](https://drive.google.com/file/d/1cNqd4DRp4_Q1ayEYz7a-_WokA0X2PXsN/view?usp=sharing).  
To use XProphetNet:  
1) Preprocess. Tokenize the text with provided bpe model and generate the binary file for Fairseq.  
2) Finetune. Finetune the model with the supervised corpus, eg English News Title Generation or English Question Generation.
3) Inference. Use finetuned checkpoint with less epoches to generate outputs for zero-shot languages, and checkpoint with more epoches for supervised language. According to the performance on dev set, for News Title Generation, we select the 18-epoch-finetuned checkpoint and 4-epoch-finetuned checkpoint for English and other zero-shot languages respectively. For Question Generation, 16 epoch for supervised English corpus and 6 epoch for zero-shoft languages.    


## Dependency

pip install torch==1.3.0  
pip install fairseq==v0.9.0  
pip install tensorboardX  
pip install sentencepiece  
pip install sacrebleu


## Preprocess
Download the [XGLUE dataset](https://microsoft.github.io/XGLUE/), uncompress, and put cross-lingual generation NTG and QG folders under ./finetune_data.  
[xprophetnet_preprocess.sh](https://github.com/microsoft/ProphetNet/blob/master/xProphetNet/xprophetnet_preprocess.sh) will 1) tokenize with the given bpe model. 2) generate the desired binary files.  
It should be noticed that although XProphetNet support maximum 512 input and output tokens, first 256 tokens are kept to be consistent with other baseline models.  
After preprocess, three types of files will appear under ./finetune_data folder:  
1) NTG folder and QG folder which contain the original text. 
2) NTG_tokenized and QG_tokenized which contain the tokenized text.  
3) NTG_processed_* and QG_processed_* which contain the binary files for Fairseq to finetune and inference.

## Finetune
With the preprocessed files under ./finetune_data and the pretrained checkpoint under ./models/ , use [finetune_xprophetnet_qg.sh](https://github.com/microsoft/ProphetNet/blob/master/xProphetNet/finetune_xprophetnet_qg.sh) and [finetune_xprophetnet_ntg.sh](https://github.com/microsoft/ProphetNet/blob/master/xProphetNet/finetune_xprophetnet_ntg.sh) to finetune XProphetNet.  
It should be noticed that the finetuning batch size is 1024. 
--update-freq means accumulate this steps to update the parameters. 
--max-sentences means maximum samples per GPU. 
Thus the batch size will be the product of update_freq, max_sentences and the number of your GPU.  
The model is finetuned on four 24GB P40 GPU.

## Inference
Inference your model with [inference_qg.sh](https://github.com/microsoft/ProphetNet/blob/master/xProphetNet/inference_qg.sh) and [inference_ntg.sh](https://github.com/microsoft/ProphetNet/blob/master/xProphetNet/finetune_xprophetnet_ntg.sh) which generate the outputs and evaluate with the original text.  
Four types of files will be generated under ./outputs folder.  
1) output_NTG_xxxxx.txt: The original outputs from Fairseq-generate.  
2) sort_hypo_NTG_xxxxxx.txt: The sorted hypothesis generated from XProphetNet, which is extracted from the former file.  
3) sort_hypo_NTG_xxxxxx.txt.post: The post-processed hypothesis file, which is recovered from the former file with bpe model
4) score_NTG_xxxxx.txt: The BLEU score  calculated with the former file and the original raw text.

It should be noticed that to generate outputs for given input, less epoches finetuned checkpoint will be used for zero-shot language,
and more epoches finetuned checkpoint will be used for the supervised language input.  
For example, xprophetnet_qg_en is finetuned on English Question Gneration data, 
checkpoint16.pt is used to generate outputs for the supervised language english input, 
while checkpoint6.pt is used to generate outputs for other zero-shot language input.

