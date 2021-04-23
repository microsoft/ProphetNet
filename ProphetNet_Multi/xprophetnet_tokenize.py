import sentencepiece as spm
import os

sp = spm.SentencePieceProcessor()
sp.Load("prophetnet_multi_dict/sentencepiece.bpe.model")

dirs = os.listdir('finetune_data/NTG')
for file_name in dirs:
	f = open('finetune_data/NTG/{}'.format(file_name), 'r', encoding='utf-8')
	file_name_part = file_name.split('.')
	file_out_name = '{}.{}.{}'.format(file_name_part[2], file_name_part[-1],file_name_part[-2])
	fout = open('finetune_data/NTG_tokenized/{}'.format(file_out_name), 'w', encoding='utf-8')
	for line in f:
		tok = sp.EncodeAsPieces(line.strip())[:256]
		fout.write('{}\n'.format(" ".join(tok)))
		

dirs = os.listdir('finetune_data/QG')
for file_name in dirs:
	f = open('finetune_data/QG/{}'.format(file_name), 'r', encoding='utf-8')
	file_name_part = file_name.split('.')
	file_out_name = '{}.{}.{}'.format(file_name_part[2], file_name_part[-1],file_name_part[-2])
	fout = open('finetune_data/QG_tokenized/{}'.format(file_out_name), 'w', encoding='utf-8')
	for line in f:
		tok = sp.EncodeAsPieces(line.strip())[:256]
		fout.write('{}\n'.format(" ".join(tok)))