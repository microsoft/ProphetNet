
# download DailyDialog & DSTC7_AVSD & PersonaChat dataset for dialogue system fine-tuning

output_file_path=./downstream_data

if [ ! -d ${output_file_path} ]; then
  mkdir -p ${output_file_path}
fi

echo 'downloading data ...'

if [ ! -f ${output_file_path}/data.tar.gz ]; then
  wget https://baidu-nlp.bj.bcebos.com/PLATO/data.tar.gz -P ${output_file_path}
fi

tar -zxvf ${output_file_path}/data.tar.gz -C ${output_file_path}/ && rm ${output_file_path}/data.tar.gz

echo 'making dir & moving data...'

dailydialog_raw_path=${output_file_path}/dailydialog/original_data
dstc7avsd_raw_path=${output_file_path}/dstc7avsd/original_data
personachat_raw_path=${output_file_path}/personachat/original_data

if [ ! -d ${dailydialog_raw_path} ]; then
  mkdir -p ${dailydialog_raw_path}
fi

if [ ! -d ${dstc7avsd_raw_path} ]; then
  mkdir -p ${dstc7avsd_raw_path}
fi

if [ ! -d ${personachat_raw_path} ]; then
  mkdir -p ${personachat_raw_path}
fi

mv ${output_file_path}/data/DailyDialog/* ${dailydialog_raw_path}
mv ${output_file_path}/data/DSTC7_AVSD/* ${dstc7avsd_raw_path}
mv ${output_file_path}/data/PersonaChat/* ${personachat_raw_path}

rm -rf ${output_file_path}/data

