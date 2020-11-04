DATASET=$1
VERSION=$2
MODEL=$3
SET=$4
PROPHETNET=prophetnet
PROPHETNET_BASE=prophetnet_base
LSTM=lstm
TRANSFORMER=transformer

if [ $MODEL = $PROPHETNET ]
then
echo 'train' $PROPHETNET
./train_prophetnet.sh $DATASET $VERSION
echo 'test' $PROPHETNET
./test_prophetnet.sh $DATASET $VERSION $SET
elif [ $MODEL = $PROPHETNET_BASE ]
then
echo $PROPHETNET_BASE
echo 'train' $PROPHETNET_BASE
./train_prophetnet_base.sh $DATASET $VERSION
echo 'test' $PROPHETNET_BASE
./test_prophetnet_base.sh $DATASET $VERSION $SET
elif [ $MODEL = $LSTM ]
then
echo $LSTM
echo 'train' $LSTM
./train_lstm.sh $DATASET $VERSION
echo 'test' $LSTM
./test_lstm.sh $DATASET $VERSION $SET
elif [ $MODEL = $TRANSFORMER ]
then
echo $TRANSFORMER
echo 'train' $TRANSFORMER
./train_transformer.sh $DATASET $VERSION
echo 'test' $TRANSFORMER
./test_transformer.sh $DATASET $VERSION $SET
else
echo 'please choose model from 1 of 4 baselines: lstm, transformer, prophetnet, prophetnet_base'
fi