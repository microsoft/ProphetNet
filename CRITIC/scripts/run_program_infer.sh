set -ex

MODEL="text-davinci-003"
# MODEL="gpt-3.5-turbo"

DATA="gsm8k"
SPLIT="test"
NUM_TEST_SAMPLE=-1

PROMPT_TYPE="pot"
# PROMPT_TYPE="direct"

TEMPERATURE=0

mkdir -p logs/$MODEL/$DATA

for START in 0
# for START in 0 300 600 900 1200
do
# END=$(expr 300 + $START)
END=-1

nohup python -um src.program.inference \
    --model $MODEL \
    --data $DATA \
    --split $SPLIT \
    --prompt_type $PROMPT_TYPE \
    --num_test_sample $NUM_TEST_SAMPLE \
    --seed 0 \
    --temperature $TEMPERATURE \
    --start $START \
    --end $END \
> logs/$MODEL/$DATA/${SPLIT}_${PROMPT_TYPE}_${NUM_TEST_SAMPLE}_t${TEMPERATURE}_s${START}_e${END}.log 2>&1&

done