DATASET=$1
VERSION=$2
SET=$3
MODEL=lstm
CNNDM=cnndm
DATA_DIR=../data/$VERSION/$DATASET\_data/processed_translation
SAVE_DIR=../outputs/$VERSION/$MODEL/$DATASET\_$MODEL
mkdir -p $SAVE_DIR



echo $DATA_DIR
echo $SAVE_DIR


if [ $DATASET = $CNNDM ]
then
BEAM=5
LENPEN=1.2
SUFFIX=_ck_best_pelt$LENPEN\_$SET\_beam$BEAM
CHECK_POINT=../models/$VERSION/$MODEL/$DATASET\_$MODEL\_checkpoints/checkpoint_best.pt
OUTPUT_FILE=$SAVE_DIR/output$SUFFIX.txt
SCORE_FILE=$SAVE_DIR/score$SUFFIX.txt

PYTHONIOENCODING=utf8 fairseq-generate $DATA_DIR --path $CHECK_POINT --task translation --truncate-source --max-source-positions 512 --min-len 45 --max-len-b 110 --no-repeat-ngram-size 3 --batch-size 32 --gen-subset $SET --beam $BEAM --num-workers 4 --lenpen $LENPEN 2>&1 > $OUTPUT_FILE
grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > $SCORE_FILE
else
BEAM=4
LENPEN=1.0
SUFFIX=_ck_best_pelt$LENPEN\_$SET\_beam$BEAM
CHECK_POINT=../models/$VERSION/$MODEL/$DATASET\_$MODEL\_checkpoints/checkpoint_best.pt
OUTPUT_FILE=$SAVE_DIR/output$SUFFIX.txt
SCORE_FILE=$SAVE_DIR/score$SUFFIX.txt

PYTHONIOENCODING=utf8 fairseq-generate $DATA_DIR --path $CHECK_POINT --task translation --truncate-source --max-source-positions 512 --max-target-positions 140 --no-repeat-ngram-size 3 --batch-size 40 --gen-subset $SET --beam $BEAM --num-workers 4 --lenpen $LENPEN 2>&1 > $OUTPUT_FILE
grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > $SCORE_FILE
fi

PYTHONIOENCODING=utf8 python eval.py --version $VERSION --dataset $DATASET --generated $SCORE_FILE

