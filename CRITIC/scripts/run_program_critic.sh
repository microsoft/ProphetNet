set -ex

MODEL="text-davinci-003"
# MODEL="gpt-3.5-turbo"

DATA="gsm8k"
SPLIT="test"
PROMPT_TYPE="pot"
NUM_TEST_SAMPLE=-1

TEMPERATURE=0.5
CRITIC_TYPE=critic
USE_TOOL=true

# CRITIC_TYPE=critic_no-tool
# USE_TOOL=false

for START in 0
# for START in 0 300 600 900 1200
do
# END=$(expr 300 + $START)
END=-1

nohup python -um src.program.critic \
    --model $MODEL \
    --data $DATA \
    --split $SPLIT \
    --prompt_type $PROMPT_TYPE \
    --critic_type $CRITIC_TYPE \
    --use_tool $USE_TOOL \
    --num_test_sample $NUM_TEST_SAMPLE \
    --seed 0 \
    --start $START \
    --end $END \
    --temperature $TEMPERATURE \
> logs/$MODEL/$DATA/${SPLIT}_${PROMPT_TYPE}_${CRITIC_TYPE}_${NUM_TEST_SAMPLE}_t${TEMPERATURE}_s${START}_e${END}.log 2>&1&

done
