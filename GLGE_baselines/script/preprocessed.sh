DATASET=$1
VERSION=$2

ORG_DATA_DIR=../data/$VERSION/$DATASET\_data/org_data
DATA_DIR_UNCASED=../data/$VERSION/$DATASET\_data/uncased_tok_data
DATA_OUT_DIR=../data/$VERSION/$DATASET\_data
echo 'tokenize dataset' $DATASET $VERSION '...'
echo 'ORG_DATA_DIR:' $ORG_DATA_DIR
echo 'DATA_DIR_UNCASED:' $DATA_DIR_UNCASED
echo 'DATA_OUT_DIR:' $DATA_OUT_DIR
mkdir -p $DATA_DIR_UNCASED

python uncase_tokenize_data.py --version $VERSION --dataset $DATASET

echo 'preprocess dataset' $DATASET $VERSION '...'

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation \
--source-lang src --target-lang tgt \
--trainpref $DATA_DIR_UNCASED/train --validpref $DATA_DIR_UNCASED/dev --testpref $DATA_DIR_UNCASED/test \
--destdir $DATA_OUT_DIR/processed_translation --srcdict ../vocab.txt --tgtdict ../vocab.txt \
--workers 20

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref $DATA_DIR_UNCASED/train --validpref $DATA_DIR_UNCASED/dev --testpref $DATA_DIR_UNCASED/test \
--destdir $DATA_OUT_DIR/processed_translation_prophetnet --srcdict ../vocab.txt --tgtdict ../vocab.txt \
--workers 20
