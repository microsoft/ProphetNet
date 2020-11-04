DATASET=$1
VERSION=$2
MODEL=lstm

DATA_DIR=../data/$VERSION/$DATASET\_data/processed_translation
SAVE_DIR=../models/$VERSION/$MODEL/$DATASET\_$MODEL\_checkpoints
mkdir -p $SAVE_DIR
ARCH=lstm
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) 
UPDATE_FREQ=$((4/$GPU_NUM))

echo $DATA_DIR
echo $SAVE_DIR
echo 'GPU' $GPU_NUM


PYTHONIOENCODING=utf8 fairseq-train \
$DATA_DIR \
--fp16 \
--task translation --arch $ARCH \
--max-sentences 64 \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.0001 \
--min-lr 1e-09 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 \
--weight-decay 0.01 \
--update-freq $UPDATE_FREQ \
--num-workers 8 \
--truncate-source \
--ddp-backend=no_c10d --max-epoch 100 \
--max-source-positions 400 --max-target-positions 120 \
--skip-invalid-size-inputs-valid-test \
--seed 1 \
--save-dir $SAVE_DIR \
--keep-last-epochs 1 \

