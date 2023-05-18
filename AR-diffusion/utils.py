import torch
import logging
import collections


logger = logging.getLogger(__name__)
CheckpointState = collections.namedtuple(
    "CheckpointState", ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])
    
    
def load_states_from_checkpoint(model_file, local_rank) -> CheckpointState:
    if local_rank == 0:
        logger.info(f'Reading saved model from {model_file}')
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    if local_rank == 0:
        logger.info(f'model_state_dict keys {state_dict.keys()}')
    return CheckpointState(**state_dict)


def get_group_parameters(config, model):
    params = list(model.named_parameters())
    no_decay = ['bias,','LayerNorm']
    other = ['time_trans']
    if config.learn_pos:
        other += ['Learned_sp']
    if config.emb_type == 'random':
        other += ['word_embedding', 'input_trans', 'output_trans']
    no_main = no_decay + other

    # weight_decay TODO
    param_group = [
        {'params':[p for n,p in params if not any(nd in n for nd in no_main)],'weight_decay':0, 'lr':1e-4},
        {'params':[p for n,p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':1e-4},
        {'params':[p for n,p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':2e-4},
        {'params':[p for n,p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':2e-4},
    ]
    return param_group