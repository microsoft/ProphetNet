#!/bin/bash  

i=6
LANG=es
BEAM=10
LENPEN=1.0
SUBSET=test
GOLDEN=xglue.qg.${LANG}.tgt.test

SUFFIX=_QG_${LANG}_ck${i}_beam${BEAM}_lp${LENPEN}_${SUBSET}_en_350k_uncased_1e5_bs1k
CHECK_POINT=./models/xprophetnet_qg_en/checkpoint${i}.pt
OUTPUT_FILE=outputs/output${SUFFIX}.txt
SCORE_FILE=outputs/score${SUFFIX}.txt

PYTHONIOENCODING=utf8 fairseq-generate finetune_data/QG_processed_${LANG} --path ${CHECK_POINT} --user-dir ./prophetnet --task translation_prophetnet --batch-size 16 --gen-subset ${SUBSET} --beam ${BEAM} --num-workers 4 --no-repeat-ngram-size 3 --lenpen $LENPEN  2>&1 > ${OUTPUT_FILE}

grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3-  > outputs/sort_hypo${SUFFIX}.txt

python post_process.py outputs/sort_hypo${SUFFIX}.txt outputs/sort_hypo${SUFFIX}.txt.post
python -m sacrebleu -lc -l ${LANG}-${LANG} ./finetune_data/QG/${GOLDEN} < outputs/sort_hypo${SUFFIX}.txt.post > ${SCORE_FILE}

