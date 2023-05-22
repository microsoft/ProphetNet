set -ex

MODEL="text-davinci-003"
# MODEL="gpt-3.5-turbo"

# DATA="ambig_qa"
# DATA="trivia_qa"
DATA="hotpot_qa"

# SPLIT="train"
SPLIT="validation"

PROMPT_TYPE="cot"
METHOD="critic"
USE_TOOL=true

# METHOD="critic_no-tool"
# USE_TOOL=false

TEMPERATURE=0
SEED=0

for START in 0
# for START in 0 100 200 300 400
do
# END=$(expr 100 + $START)
END=500

mkdir -p logs/$MODEL/$DATA

nohup python -um src.qa.critic \
    --model $MODEL \
    --data $DATA \
    --critic_type $METHOD \
    --split $SPLIT \
    --prompt_type $PROMPT_TYPE \
    --use_tool $USE_TOOL \
    --num_test_sample 500 \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --start $START \
    --end $END \
> logs/$MODEL/$DATA/${METHOD}_${SPLIT}_${PROMPT_TYPE}_${METHOD}_tools-${USE_TOOL}_seed${SEED}_t${TEMPERATURE}_s${START}_e${END}.log 2>&1&

done