
import os
import sys
sys.path.append('/home/v-wchen2/PycharmProjects/ProphetNet')

from src.utils.processor import FINETUNE_PREFIX_PATH, dialog_input_output_len, convert_daily_dialog, check


dialog_input_output_len(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/original_data/dial.train'), has_knowledge=False)
dialog_input_output_len(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/original_data/dial.valid'), has_knowledge=False)


if not os.path.exists(os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed')):
    os.makedirs(os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed'))

convert_daily_dialog(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/original_data/dial.train'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed/train.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed/train.tgt'),
    test=False,
)
convert_daily_dialog(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/original_data/dial.valid'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed/valid.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed/valid.tgt'),
    test=False,
)
convert_daily_dialog(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/original_data/dial.test'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed/test.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed/test.tgt'),
    test=True,
)

check(processed_path=os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog/processed'))

