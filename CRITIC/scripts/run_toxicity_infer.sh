set -ex

DATA="toxicity"

MODEL="text-davinci-003"
# MODEL="gpt-3.5-turbo"

SPLIT="test"

START=0
END=-1
TEMPERATURE=0.9

mkdir -p logs/$MODEL/$DATA

nohup python -um src.toxicity.inference \
    --model $MODEL \
    --data $DATA \
    --split $SPLIT \
    --seed 0 \
    --start $START \
    --end $END \
    --temperature $TEMPERATURE \
> logs/$MODEL/$DATA/${SPLIT}_s${START}_e${END}_t${TEMPERATURE}.log 2>&1&

