set -ex

DATA="toxicity"

MODEL="text-davinci-003"
# MODEL="gpt-3.5-turbo"

SPLIT="test"

TEMPERATURE=0.9
CRITIC_TYPE=critic
USE_TOOL=true

# CRITIC_TYPE=critic_v1_no-tool
# USE_TOOL=false

mkdir -p logs/$MODEL/$DATA


for START in 0
do
# END=$(expr 100 + $START)
END=-1

nohup python -um src.toxicity.critic \
    --model $MODEL \
    --data $DATA \
    --split $SPLIT \
    --critic_type $CRITIC_TYPE \
    --use_tool $USE_TOOL \
    --seed 0 \
    --start $START \
    --end $END \
    --temperature $TEMPERATURE \
> logs/$MODEL/$DATA/${SPLIT}_critic_tools-${USE_TOOL}_s${START}_e${END}_t${TEMPERATURE}.log 2>&1&

done