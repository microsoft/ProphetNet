#!/bin/bash  

i=4
LANG=es
BEAM=10
LENPEN=1.0
SUBSET=test
GOLDEN=xglue.ntg.${LANG}.tgt.test

SUFFIX=_NTG_${LANG}_ck${i}_beam${BEAM}_lp${LENPEN}_${SUBSET}_es_350k_uncased_1e5_bs1k
CHECK_POINT=./models/xprophetnet_ntg_en/checkpoint${i}.pt
OUTPUT_FILE=outputs/output${SUFFIX}.txt
SCORE_FILE=outputs/score${SUFFIX}.txt

PYTHONIOENCODING=utf8 fairseq-generate finetune_data/NTG_processed_${LANG} --path ${CHECK_POINT} --user-dir ./prophetnet --task translation_prophetnet --batch-size 16 --gen-subset ${SUBSET} --beam ${BEAM} --num-workers 4 --no-repeat-ngram-size 3 --lenpen $LENPEN  2>&1 > ${OUTPUT_FILE}

grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- > outputs/sort_hypo${SUFFIX}.txt

python post_process.py outputs/sort_hypo${SUFFIX}.txt outputs/sort_hypo${SUFFIX}.txt.post
python -m sacrebleu -lc -l ${LANG}-${LANG} ./finetune_data/NTG/${GOLDEN} < outputs/sort_hypo${SUFFIX}.txt.post > ${SCORE_FILE}


