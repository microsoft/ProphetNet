DATASET=$1
VERSION=$2
MODEL=prophetnet
CNNDM=cnndm
DATA_DIR=../data/$VERSION/$DATASET\_data/processed_translation_prophetnet
USER_DIR=./prophetnet
SAVE_DIR=../models/$VERSION/$MODEL/$DATASET\_$MODEL\_checkpoints
mkdir -p $SAVE_DIR
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
PRETRAINED_MODEL=../pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt
GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) 

echo $DATA_DIR
echo $SAVE_DIR
echo 'GPU' $GPU_NUM


if [ $DATASET = $CNNDM ] 
then
MAX_SENTENCE=2
UPDATE_FREQ=$((32*16/$MAX_SENTENCE/$GPU_NUM))
PYTHONIOENCODING=utf8 
fairseq-train $DATA_DIR \
	--fp16 \
	--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
	--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
	--lr 0.0001 --min-lr 1e-09 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
	--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	--criterion $CRITERION --label-smoothing 0.1 \
	--update-freq $UPDATE_FREQ --max-sentences $MAX_SENTENCE \
	--num-workers 8  \
	--load-from-pretrained-model $PRETRAINED_MODEL \
	--ddp-backend=no_c10d --max-epoch 10 \
	--max-source-positions 512 --max-target-positions 512 \
	--skip-invalid-size-inputs-valid-test \
	--seed 1 \
	--save-dir $SAVE_DIR \
	--keep-last-epochs 1
else
UPDATE_FREQ=$((8/$GPU_NUM))
PYTHONIOENCODING=utf8 
fairseq-train $DATA_DIR \
	--fp16 \
	--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
	--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
	--lr 0.0001 --min-lr 1e-09 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
	--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	--criterion $CRITERION --label-smoothing 0.1 \
	--update-freq $UPDATE_FREQ --max-tokens 1300 --max-sentences 16 \
	--num-workers 8  \
	--load-from-pretrained-model $PRETRAINED_MODEL \
	--ddp-backend=no_c10d --max-epoch 10 \
	--max-source-positions 512 --max-target-positions 512 \
	--skip-invalid-size-inputs-valid-test \
	--seed 1 \
	--save-dir $SAVE_DIR \
	--keep-last-epochs 1
fi


