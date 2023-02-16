DATA_DIR=data/processed_stc_dialog/
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=models/finetune_stc_dialog
TENSORBOARD_LOGDIR=models/tensorboard_stc_dialog
USER_DIR=./prophetnet
PRETRAINED_CHECKPOINT=./pretrained_checkpoint/checkpoint_dialog_zh.pt

python /home/v-weizqi/coqa/fairseq/train.py $DATA_DIR \
	--fp16 --ngram 2 \
	--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
	--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
	--lr 0.0001 --min-lr 1e-09 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
	--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	--criterion $CRITERION --label-smoothing 0.1 \
	--update-freq 8 --max-sentences 16 \
	--num-workers 8  \
	--ddp-backend=no_c10d --max-epoch 10 \
	--max-source-positions 512 --max-target-positions 512 \
	--truncate-source --load-from-pretrained-model $PRETRAINED_CHECKPOINT \
	--save-dir $SAVE_DIR \
	--keep-last-epochs 10 \
	--tensorboard-logdir $TENSORBOARD_LOGDIR 
 
