PROJECT_ROOT=/vc_data/users/`whoami`/codeProphetnet
DATA_DIR_ROOT=$PROJECT_ROOT/data/processed_code2text
SAVE_DIR_ROOT=$PROJECT_ROOT/models/finetune_code2text
TENSORBOARD_LOGDIR_ROOT=$PROJECT_ROOT/models/tensorboard_code2text

ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
USER_DIR=./prophetnet
CHECKPOINT=45
PRETRAINED_CHECKPOINT=$PROJECT_ROOT/pretrain_checkpoints/checkpoint$CHECKPOINT.pt

LR=0.0001
ATTENTION_DROP_OUT=0.1
DROP_OUT=0.1
DECAY=0.01
LS=0.1


#LR=5e-5
#DROP_OUT=0.1
#ATTENTION_DROP_OUT=0.1
#DECAY=0.0
#LS=0.1

#for lang in go; do
#for lang in medium small; do
for lang in go python ruby javascript php java; do
	DATA_DIR="${DATA_DIR_ROOT}/processed_$lang"
	SAVE_DIR=${SAVE_DIR_ROOT}/${CHECKPOINT}/$lang
	TENSORBOARD_LOGDIR=${TENSORBOARD_LOGDIR_ROOT}/${CHECKPOINT}/$lang
	mkdir -p $SAVE_DIR
	mkdir -p $TENSORBOARD_LOGDIR
	fairseq-train \
		$DATA_DIR \
		--fp16 --ngram 2 \
		--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
		--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
		--lr $LR --min-lr 1e-09 \
		--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
		--dropout $DROP_OUT --attention-dropout $ATTENTION_DROP_OUT --weight-decay $DECAY \
		--criterion $CRITERION --label-smoothing $LS \
		--update-freq 4 --max-sentences 6 \
		--num-workers 8  \
		--ddp-backend=no_c10d --max-epoch 10 \
		--max-source-positions 512 --max-target-positions 512 \
		--truncate-source --load-from-pretrained-model $PRETRAINED_CHECKPOINT \
		--save-dir $SAVE_DIR \
		--keep-last-epochs 10 \
		--tensorboard-logdir $TENSORBOARD_LOGDIR 
done