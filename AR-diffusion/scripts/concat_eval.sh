export PYTHONPATH=$PYTHONPATH:./AR-Diffusion

FILE_NAME=cnndm
DATA_NAME=cnn_dm
STEP=620000
NUM=50

echo "model step" $STEP
j=0
while [ "$j" -lt $NUM ]; do
echo "gen num $j"
./.conda/envs/torch/bin/python ./eval_utils/concat.py \
--n_gpu=8 --num=$j \
--src_path=./my_output/$DATA_NAME/$FILE_NAME/$STEP\_ema_0.99_skip__xy_20/num$j \
--tgt_path=./data/$DATA_NAME

# _ema_0.99

j=$(($j+1))
done


./.conda/envs/torch/bin/python ./eval_utils/mbr/mbr_select.py \
--data_name=$DATA_NAME --num=$NUM --process=50 # --exp_name=500


# IWSLT14
# dinosier
# ./.conda/envs/torch/bin/python ./eval_utils/iwslt/eval.py --pred ./data/iwslt14/gen_.txt --lang de --ref ./data/iwslt14/test.de
# ./.conda/envs/torch/bin/python ./eval_utils/iwslt/eval.py --pred ./data/iwslt14/gen_.txt --lang en --ref ./data/iwslt14/test.en

# seqdiffseq
# ./.conda/envs/torch/bin/python ./eval_utils/iwslt/eval.py --pred ./data/iwslt14_tok/gen_.txt --lang en --ref ./data/iwslt14_tok/test.en 


# XSUM
# source activate rouge
# files2rouge ./data/xsum/gen_.txt ./data/xsum/test.tgt


# CNN_DM
# ./.conda/envs/rouge/bin/python ./eval_utils/cnndm/postprocess_cnn_dm.py --generated ./data/cnn_dm/gen_.txt --golden ./data/cnn_dm/test.tgt


# Commongen
# echo "Start running ROUGE"
# ./.conda/envs/rouge/bin/python ./eval_utils/rouge/eval.py --pred ${PRED_FILE}   --gold ${TRUTH_FILE} --perl

# echo "BLEU/METER/SPICE"
# ./.conda/envs/eval/bin/python ./eval_utils/commongen/eval.py --key_file ${INPUT_FILE} --gts_file ${TRUTH_FILE} --res_file ${PRED_FILE}


# Diversity (Self-Bleu)
# python ./eval_utils/diversity.py --data=iwslt14
