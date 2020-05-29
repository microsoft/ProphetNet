mkdir ./finetune_data/NTG_tokenized
mkdir ./finetune_data/QG_tokenized

python xprophetnet_tokenize.py

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref ./finetune_data/QG_tokenized/en.train --validpref ./finetune_data/QG_tokenized/en.dev --testpref ./finetune_data/QG_tokenized/en.test \
--destdir ./finetune_data/QG_processed_en --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/QG_tokenized/de.dev --testpref ./finetune_data/QG_tokenized/de.test \
--destdir ./finetune_data/QG_processed_de --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15


fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/QG_tokenized/es.dev --testpref ./finetune_data/QG_tokenized/es.test \
--destdir ./finetune_data/QG_processed_es --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/QG_tokenized/fr.dev --testpref ./finetune_data/QG_tokenized/fr.test \
--destdir ./finetune_data/QG_processed_fr --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/QG_tokenized/it.dev --testpref ./finetune_data/QG_tokenized/it.test \
--destdir ./finetune_data/QG_processed_it --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/QG_tokenized/pt.dev --testpref ./finetune_data/QG_tokenized/pt.test \
--destdir ./finetune_data/QG_processed_pt --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15


fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref ./finetune_data/NTG_tokenized/en.train --validpref ./finetune_data/NTG_tokenized/en.dev --testpref ./finetune_data/NTG_tokenized/en.test \
--destdir ./finetune_data/NTG_processed_en --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/NTG_tokenized/fr.dev --testpref ./finetune_data/NTG_tokenized/fr.test \
--destdir ./finetune_data/NTG_processed_fr --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/NTG_tokenized/es.dev --testpref ./finetune_data/NTG_tokenized/es.test \
--destdir ./finetune_data/NTG_processed_es --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/NTG_tokenized/de.dev --testpref ./finetune_data/NTG_tokenized/de.test \
--destdir ./finetune_data/NTG_processed_de --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--validpref ./finetune_data/NTG_tokenized/ru.dev --testpref ./finetune_data/NTG_tokenized/ru.test \
--destdir ./finetune_data/NTG_processed_ru --srcdict xlmr_dictionary/dict.txt --tgtdict xlmr_dictionary/dict.txt \
--workers 15

