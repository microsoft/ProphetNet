## 💬 Reference

- *XSum*: [here](https://github.com/microsoft/ProphetNet/tree/master/GLGE_baselines/script/script/evaluate)
- *CNN/DailyMail*: [here](https://github.com/microsoft/ProphetNet/tree/master/GLGE_baselines/script/script/evaluate)
- *IWSLT14*: ```eval_utils/iwslt/```
- *Commongen*: [here](https://github.com/INK-USC/CommonGen/tree/master/evaluation/Traditional/eval_metrics)


## 🚀 rouge
```
1. conda create -n rouge python=3.6
2. source activate rouge
3. git clone https://github.com/bheinzerling/pyrouge
4. cd pyrouge
5. pip install -e .
6. git clone https://github.com/andersjo/pyrouge.git rouge
7. pyrouge_set_rouge_path ~/pyrouge/rouge/tools/ROUGE-1.5.5/  （use "pip show pyrouge" to change to absolute path）
8. sudo apt-get install libxml-parser-perl
9. cd rouge/tools/ROUGE-1.5.5/data
10. rm WordNet-2.0.exc.db
11. ./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
    (
        if perl warnings: 
        "sudo apt-get install -y locales"
        "sudo locale-gen en_US.UTF-8"
    )
12. python -m pyrouge.test
13. pip install tqdm numpy py-rouge nltk
```

```python
import nltk
nltk.download('punkt')
```

### files2rouge
```
1. pip install -U git+https://github.com/pltrdy/pyrouge
2. git clone https://github.com/pltrdy/files2rouge.git     
3. cd files2rouge
4. python setup_rouge.py
5. python setup.py install
```

## ⚙️ spice
```
1. conda create -n eval python=2.7
2. conda activate eval
3. pip install numpy
4. pip install -U spacy
5. python -m spacy download en_core_web_sm
6. sudo apt-get install zip
   "modified the file path"
7. bash get_stanford_models.sh
```
## 💡 install java
```
1. sudo cp ../jdk-8u311-linux-x64.tar.gz /opt
2. cd /opt
3. sudo mkdir java
4. sudo chown user java
5. sudo tar -zxvf jdk-8u311-linux-x64.tar.gz -C /opt/java
6. sudo vi /etc/profile
7. Append the following information to the file:

#set java environment
export JAVA_HOME=/opt/java/jdk1.8.0_311
export PATH=${JAVA_HOME}/bin:${PATH}
    
6. exit and save: source /etc/profile
7. set it up again in bashrc: sudo vi ~/.bashrc and add the same information as above and run source ~/.bashrc
8. check: java -version
```