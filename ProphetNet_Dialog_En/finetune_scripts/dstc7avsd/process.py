
import os
import sys
sys.path.append('/home/v-wchen2/PycharmProjects/ProphetNet')

from src.utils.processor import FINETUNE_PREFIX_PATH, dialog_input_output_len, convert_dstc7_avsd, check


dialog_input_output_len(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/original_data/dial.train'), has_knowledge=True)
dialog_input_output_len(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/original_data/dial.valid'), has_knowledge=True)


if not os.path.exists(os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed')):
    os.makedirs(os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed'))


convert_dstc7_avsd(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/original_data/dial.train'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed/train.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed/train.tgt'),
    test=False,
)
convert_dstc7_avsd(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/original_data/dial.valid'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed/valid.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed/valid.tgt'),
    test=False,
)
convert_dstc7_avsd(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/original_data/dial.test'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed/test.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed/test.tgt'),
    multi_ref_tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed/test_multi_refs.tgt'),
    test=True,
)

check(processed_path=os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed'))

