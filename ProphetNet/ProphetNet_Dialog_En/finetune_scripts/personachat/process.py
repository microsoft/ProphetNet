
import os
import sys
sys.path.append('/home/v-wchen2/PycharmProjects/ProphetNet')

from src.utils.processor import FINETUNE_PREFIX_PATH, dialog_input_output_len, convert_persona_chat, check


dialog_input_output_len(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/original_data/dial.train'), has_knowledge=True)
dialog_input_output_len(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/original_data/dial.valid'), has_knowledge=True)


if not os.path.exists(os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed')):
    os.makedirs(os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed'))


# PersonaChat
convert_persona_chat(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/original_data/dial.train'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed/train.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed/train.tgt'),
    test=False,
)
convert_persona_chat(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/original_data/dial.valid'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed/valid.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed/valid.tgt'),
    test=False,
)
convert_persona_chat(
    fin=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/original_data/dial.test'),
    src_fout=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed/test.src'),
    tgt_fout=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed/test.tgt'),
    test=True,
)

check(processed_path=os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed'))

