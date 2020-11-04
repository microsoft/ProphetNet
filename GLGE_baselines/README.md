# ProphetNet baselines for GLGE

This is the implement of ProphetNet [GLGE](https://arxiv.org/abs/) baseline. More details about GLGE can be found in [GLGE github repo](https://github.com/microsoft/glge) and the dataset can be downloaded from [link](https://microsoft.github.io/glge/).  

## Run Baselines

### Baseline Dependency
- pip install torch==1.3.0  
- pip install fairseq==v0.9.0

### Evaluation Dependency
- pip install git+https://github.com/yuyan2do/Distinct-N.git
- pip install git+https://github.com/pltrdy/pyrouge`
- pip install py-rouge
- pip install nltk
- install [file2rouge](https://github.com/pltrdy/files2rouge)

We also provide a docker image which contains the evaluation dependency.
`docker run -it adsbrainwestus2.azurecr.io/glge /bin/bash`

### Data Preparation
Download the datasets from [link](https://microsoft.github.io/glge/) and move them to the `data/` folder.
To preprocess all the datasets, please use the following command to tokenize the data and generate the binary data files:
```
cd script
./preprocessed-all.sh
```
Note that we tokenize each dataset with BERT-uncased tokenizer.

To preprocess a specific dataset, please use the following command:
```
./preprocessed.sh `<DATASET>` `<VERSION>`
```
For example, if you want to preprocess the easy version of CNN/DailyMail, please use the following command:
```
./preprocessed.sh cnndm easy
```
Here `<DATASET>` can be `cnndm`, `gigaword`, `xsum`, `msnews`, `squadqg`, `msqg`, `coqa`, `personachat`. `<VERSION>` can be `easy`, `medium`, `hard`.

### Checkpoints
If you want to use ProphetNet-large, please download the pretrained checkpoints at [here](https://github.com/microsoft/ProphetNet) and move them to the `pretrained_checkpoints/` folder.

### Training and Testing Pipeline
We provide 4 baselines, including LSTM, Transformer, ProphetNet-base, and ProphetNet-large basd on [fairseq](https://github.com/pytorch/fairseq).
    
To train and test the baselines, please use the following command:
```
cd script
./run.sh `<DATASET>` `<VERSION>` `<MODEL>` `<SET>`
```
For example, if you want to train and test the ProphetNet-large on the medium version of SQuAD 1.1 question generation dev set, please use the following command:
```
./run.sh squadqg medium prophetnet dev
```
Here `<MODEL>` can be `lstm`, `transformer`, `prophetnet_base`, `prophetnet`, and `<SET>` can be `dev`, `test`.
    
The checkpoints will be saved at the `models` folder and the outputs will be saved at the `outputs` folder.
