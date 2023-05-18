export PYTHONPATH=$PYTHONPATH:./AR-Diffusion

# XSum
FILE_NAME = xsum
STEP = 80000

torchrun --nproc_per_node=8 --nnodes=1 ./gen_utils/generate.py \
model.name='bert-base-uncased' batch_size=800 \
exp.name=$FILE_NAME load_step=$STEP \
data.name=xsum max_pos_len=512 num_samples=50 \
intermediate_size=2048 num_attention_heads=8 \
in_channels=128 out_channels=128 time_channels=128 \
skip_sample=True gen_timesteps=20 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' \
tgt_len=50 prediction=True load_from_ema=True \



# CNN/DailyMail
FILE_NAME = cnn_dm
STEP = 100000

torchrun --nproc_per_node=8 --nnodes=1 ./gen_utils/generate.py \
model.name='bert-base-uncased' batch_size=250 \
exp.name=$FILE_NAME load_step=$STEP \
data.name=cnn_dm max_pos_len=512 num_samples=50 \
intermediate_size=2048 num_attention_heads=8 dropout=0.2 \
in_channels=128 out_channels=128 time_channels=128 \
skip_sample=True gen_timesteps=20 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' \
tgt_len=180 prediction=True load_from_ema=True \



# IWSLT14
FILE_NAME = iwslt14
STEP = 230000

torchrun --nproc_per_node=8 --nnodes=1 ./gen_utils/generate.py \
model.name='bert-base-uncased' batch_size=800  \
exp.name=$FILE_NAME load_step=$STEP \
data.name=iwslt14_tok tgt_len=90 max_pos_len=256 num_workers=4 use_bpe=True num_samples=50 \
intermediate_size=1024 num_attention_heads=4 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
skip_sample=True gen_timesteps=20 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' \



# IWSLT14 / dinoiser
# de -> en
FILE_NAME = iwslt14_deen
STEP = 135000

torchrun --nproc_per_node=8 --nnodes=1 ./gen_utils/generate.py \
model.name='bert-base-uncased' batch_size=800  \
exp.name=$FILE_NAME load_step=$STEP \
data.name=iwslt14 tgt_len=90 max_pos_len=256 num_workers=4 use_bpe=True num_samples=50 \
intermediate_size=1024 num_attention_heads=4 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
skip_sample=True gen_timesteps=20 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' \

# en -> de
FILE_NAME = iwslt14_ende
STEP = 51000

torchrun --nproc_per_node=8 --nnodes=1 ./gen_utils/generate.py \
model.name='bert-base-uncased' batch_size=800  \
exp.name=$FILE_NAME load_step=$STEP \
data.name=iwslt14 tgt_len=90 max_pos_len=90 num_workers=4 use_bpe=True num_samples=50 \
intermediate_size=1024 num_attention_heads=4 dropout=0.2 \
in_channels=64 out_channels=64 time_channels=64 \
skip_sample=True gen_timesteps=20 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' \
src_lang='en' tgt_lang='de' \



# Commongen
FILE_NAME = commongen
STEP = 40000

torchrun --nproc_per_node=8 --nnodes=1 ./gen_utils/generate.py \
model.name='bert-base-uncased' batch_size=800 \
exp.name=$FILE_NAME load_step=$STEP \
data.name=commongen max_pos_len=128 num_samples=50 \
intermediate_size=512 num_attention_heads=8 \
in_channels=64 out_channels=64 time_channels=64 \
skip_sample=True gen_timesteps=20 \
schedule_sampler='xy_uniform' time_att=True att_strategy='txl' \
tgt_len=54 prediction=True \
