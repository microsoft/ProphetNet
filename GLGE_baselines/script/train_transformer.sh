DATASET=$1
VERSION=$2
MODEL=transformer

DATA_DIR=../data/$VERSION/$DATASET\_data/processed_translation
SAVE_DIR=../models/$VERSION/$MODEL/$DATASET\_$MODEL\_checkpoints
mkdir -p $SAVE_DIR

GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) 

echo $DATA_DIR
echo $SAVE_DIR
echo 'GPU' $GPU_NUM

ARCH=transformer_vaswani_wmt_en_de_big
CRITERION=label_smoothed_cross_entropy


MAX_SENTENCE=8
UPDATE_FREQ=$((32*16/$MAX_SENTENCE/$GPU_NUM))
fairseq-train --fp16 --task translation --arch $ARCH --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.2 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --weight-decay 0.01 --criterion $CRITERION --label-smoothing 0.1 --update-freq $UPDATE_FREQ --max-sentences $MAX_SENTENCE --truncate-source --num-workers 4 --max-epoch 20 --max-source-positions 512 --max-target-positions 512 --skip-invalid-size-inputs-valid-test --seed 1 --save-dir $SAVE_DIR --log-interval 1 --find-unused-parameters --encoder-layerdrop 0.2 --decoder-layerdrop 0.2 --memory-efficient-fp16 --find-unused-parameters $DATA_DIR --keep-last-epochs 1 --ddp-backend=no_c10d
