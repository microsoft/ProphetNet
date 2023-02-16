#!/bin/bash
# parallel -j8 "bash inference_code2text_single.sh {1} {2} {%} 39" ::: 6 7 8 9 10 ::: java php go ruby javascript python
#PROJECT_ROOT=/vc_data/users/`whoami`/code_prophetnet_finetune
# langs="ruby javascript go python java php" parallel -j1 "echo lang:{1} epoch:{2} && cat outputs/{3}/sort_hypo_e{3}_text2code_beam4_{1}_lp1.0_test_ck{2}.pred.bleu4" ::: $langs ::: 6 7 8 9 10 ::: 25
PROJECT_ROOT=/vc_data/users/`whoami`/codeProphetnet
i=$1
lang=$2

((CUDA_VISIBLE_DEVICES=$3-1))

PRETRAIN_CHECKPOINT=$4

BEAM=4
LENPEN=1.0
SUBSET=test
#DATA_DIR=./data/processed_code2text/processed_$lang
DATA_DIR=./test/$lang

SUFFIX=_e${PRETRAIN_CHECKPOINT}_code2text_beam${BEAM}_${lang}_lp${LENPEN}_${SUBSET}_ck${i}
#CHECK_POINT=./models/finetune_code2text/${PRETRAIN_CHECKPOINT}/${lang}/checkpoint${i}.pt
CHECK_POINT=$PROJECT_ROOT/models/finetune_code2text/${PRETRAIN_CHECKPOINT}/${lang}/checkpoint${i}.pt
OUTPUT_ROOT=$PROJECT_ROOT/outputs/${PRETRAIN_CHECKPOINT}
OUTPUT_FILE=$OUTPUT_ROOT/output${SUFFIX}.txt
GOLDEN_FILE=$OUTPUT_ROOT/sort_hypo${SUFFIX}.golden
PRED_FILE=$OUTPUT_ROOT/sort_hypo${SUFFIX}.pred
mkdir -p $OUTPUT_ROOT

export PYTHONIOENCODING=utf8

PYTHONIOENCODING=utf8 fairseq-generate ${DATA_DIR} --path ${CHECK_POINT} --user-dir ./prophetnet \
--task translation_prophetnet --batch-size 16 --gen-subset ${SUBSET} --beam ${BEAM} --num-workers 4 --no-repeat-ngram-size 3 \
--max-source-positions 512 --truncate-source --lenpen $LENPEN  2>&1 > ${OUTPUT_FILE}

grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- > $PRED_FILE

grep ^T $OUTPUT_FILE | cut -c 3- | sort -n | cut -f2- > $GOLDEN_FILE

python post_process.py $PRED_FILE $PRED_FILE.post

python post_process.py $GOLDEN_FILE $GOLDEN_FILE.post

python eval.py $GOLDEN_FILE.post < $PRED_FILE.post | tee $PRED_FILE.bleu4

echo $lang evaluation done.
