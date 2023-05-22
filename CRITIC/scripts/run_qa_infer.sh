set -ex

MODEL="text-davinci-003"
# MODEL="gpt-3.5-turbo"

# DATA="ambig_qa"
# DATA="trivia_qa"
DATA="hotpot_qa"

SPLIT="validation"

# PROMPT_TYPE="direct"
PROMPT_TYPE="cot"

TEMPERATURE=0
NUM_SAMPLING=1

START=0
END=-1
SEED=0

mkdir -p logs/$MODEL/$DATA

nohup python -um src.qa.inference \
    --model $MODEL \
    --data $DATA \
    --split $SPLIT \
    --prompt_type $PROMPT_TYPE \
    --num_test_sample 500 \
    --num_sampling $NUM_SAMPLING \
    --temperature $TEMPERATURE \
    --seed $SEED \
    --start $START \
    --end $END \
> logs/$MODEL/$DATA/${SPLIT}_${PROMPT_TYPE}_seed${SEED}_t${TEMPERATURE}_s${START}_e${END}.log 2>&1&
