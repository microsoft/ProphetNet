export PYTHONPATH=$PYTHONPATH:./AR-Diffusion

# XSum
FILE_NAME = xsum
STEP = 80000

torchrun --nproc_per_node=8 --nnodes=1 ./train_utils/trainer_main.py \
model.name='bert-base-uncased' batch_size=128 grad_accum=3 \
total_steps=$STEP exp.name=$FILE_NAME \
data.name=xsum tgt_len=50 max_pos_len=512 lr=8e-4 lr_step=40000 \
intermediate_size=2048 num_attention_heads=8 dropout=0.2 \
in_channels=128 out_channels=128 time_channels=128 \
eval_interval=3000 log_interval=1000 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' use_AMP=True \



# CNN/DailyMail
FILE_NAME = cnn_dm
STEP = 100000

torchrun --nproc_per_node=8 --nnodes=1 ./train_utils/trainer_main.py \
model.name='bert-base-uncased' batch_size=80 grad_accum=5 \
total_steps=$STEP exp.name=$FILE_NAME \
data.name=xsum tgt_len=180 max_pos_len=512 lr=8e-4 lr_step=30000 \
intermediate_size=2048 num_attention_heads=8 dropout=0.2 \
in_channels=128 out_channels=128 time_channels=128 \
eval_interval=3000 log_interval=1000 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' use_AMP=True \



# IWSLT14
FILE_NAME = iwslt14
STEP = 230000

torchrun --nproc_per_node=8 --nnodes=1 ./train_utils/trainer_main.py \
model.name='bert-base-uncased' batch_size=128 grad_accum=3 \
total_steps=$STEP exp.name=$FILE_NAME \
data.name=iwslt14_tok tgt_len=90 max_pos_len=256 lr=1.6e-3 num_workers=4 use_bpe=True lr_step=80000 \
intermediate_size=1024 num_attention_heads=4 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
eval_interval=3000 log_interval=1000 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' use_AMP=True \



# IWSLT14 / dinoiser
# de -> en
FILE_NAME = iwslt14_deen
STEP = 135000

torchrun --nproc_per_node=8 --nnodes=1 ./train_utils/trainer_main.py \
model.name='bert-base-uncased' batch_size=192 grad_accum=2 \
total_steps=$STEP exp.name=$FILE_NAME \
data.name=iwslt14 tgt_len=90 max_pos_len=256 lr=2e-3 num_workers=4 use_bpe=True lr_step=80000 \
intermediate_size=1024 num_attention_heads=4 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
eval_interval=3000 log_interval=1000 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' use_AMP=True \

# en -> de
FILE_NAME = iwslt14_ende
STEP = 51000

torchrun --nproc_per_node=8 --nnodes=1 ./train_utils/trainer_main.py \
model.name='bert-base-uncased' batch_size=768 grad_accum=1 \
total_steps=$STEP exp.name=$FILE_NAME \
data.name=iwslt14 tgt_len=90 max_pos_len=90 lr=1.8e-3 num_workers=4 use_bpe=True lr_step=60000 \
intermediate_size=1024 num_attention_heads=4 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
eval_interval=3000 log_interval=1000 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' use_AMP=True \
src_lang='en' tgt_lang='de' \



# Commongen
FILE_NAME = commongen
STEP = 40000

torchrun --nproc_per_node=8 --nnodes=1 ./train_utils/trainer_main.py \
model.name='bert-base-uncased' batch_size=384 grad_accum=1 \
total_steps=$STEP exp.name=$FILE_NAME \
data.name=iwslt14_tok tgt_len=54 max_pos_len=128 lr=3e-4 lr_step=40000 \
intermediate_size=512 num_attention_heads=8 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
eval_interval=3000 log_interval=1000 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' \
