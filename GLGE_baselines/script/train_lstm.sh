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


CRITERION=label_smoothed_cross_entropy


MAX_SENTENCE=8
UPDATE_FREQ=$((32*16/$MAX_SENTENCE/$GPU_NUM))
fairseq-train $DATA_DIR --task translation --arch $ARCH --truncate-source --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 --lr 0.0003 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --dropout 0.1 --weight-decay 0.01 --criterion $CRITERION --label-smoothing 0.1 --update-freq $UPDATE_FREQ --max-sentences $MAX_SENTENCE --truncate-source --num-workers 4 --max-epoch 100 --max-source-positions 512 --max-target-positions 512 --seed 1 --save-dir $SAVE_DIR --keep-last-epochs 1 --ddp-backend=no_c10d

