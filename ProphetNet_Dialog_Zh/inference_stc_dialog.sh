#!/bin/bash  


i=7
BEAM=4
LENPEN=1.0
SUBSET=test
DATA_DIR=./data/processed_stc_dialog/

SUFFIX=_stc_dialog_beam${BEAM}_lp${LENPEN}_${SUBSET}_ck${i}
CHECK_POINT=./models/finetune_stc_dialog/checkpoint${i}.pt
OUTPUT_FILE=outputs/output${SUFFIX}.txt


PYTHONIOENCODING=utf8 python /home/v-weizqi/coqa/fairseq/generate.py ${DATA_DIR} --path ${CHECK_POINT} --user-dir ./prophetnet --task translation_prophetnet --batch-size 16 --gen-subset ${SUBSET} --beam ${BEAM} --num-workers 4 --no-repeat-ngram-size 3 --lenpen $LENPEN  2>&1 > ${OUTPUT_FILE}

grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3-  | sed "s/ ##//g" > outputs/sort_hypo${SUFFIX}.txt
