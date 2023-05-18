import logging

import torch.distributed as dist
import model_utils.gaussian_diffusion as gd

from model_utils.gaussian_diffusion import GaussianDiffusion
from model_utils.diffusion_lm import CrossAttention_Diffusion_LM

logger = logging.getLogger(__name__)

def create_model(config, vocab_size):
    if dist.get_rank() == 0:
        logger.info(f'creating model, based on {config.model.mode}')
        logger.info(f'loading bart {config.load_bart}')
        logger.info(f'rescaling timesteps {config.rescale_timesteps}')
        logger.info(f'using self condition {config.self_condition}')
        logger.info(f'learning time position {config.learn_pos}')
        logger.info(f'fixing encoder {config.fix_encoder}')

    if config.model.mode == 's2s':
        return CrossAttention_Diffusion_LM(
            config=config,
            vocab_size = vocab_size,
            out_channels=(config.out_channels if not config.learn_sigma else config.out_channels * 2),
        )
    else:
        raise NotImplementedError


def create_gaussian_diffusion(config):

    # The value of β is determined according to the maximum T and variance schedule strategy
    betas = gd.get_named_beta_schedule(
        config.noise_schedule, config.diffusion_steps)
    if dist.get_rank() == 0:
        logger.info(f"noise_schedule: {config.noise_schedule}")
        logger.info(f"diffusion steps: {config.diffusion_steps}")
        logger.info(f"betas: {betas}")

    # Decide the loss function to be used for training
    if config.model.mode == 'e2e' or config.model.mode == 's2s':
        # end to end training
        if config.use_kl:
            loss_type = gd.LossType.E2E_KL
        else:
            loss_type = gd.LossType.E2E_MSE  # choose
    elif config.model.mode == 'e2e-simple':
        if config.use_kl:
            loss_type = gd.LossType.E2E_Simple_KL
        else:
            loss_type = gd.LossType.E2E_Simple_MSE
    else:
        if config.use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif config.rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE

    if dist.get_rank() == 0:
        logger.info(f"Diffusion Loss Type: {loss_type}")
        logger.info(f"Whether to learn sigma: {config.learn_sigma}")
        logger.info(f"Diffusion predict xstart: {config.predict_xstart}")
        logger.info(f"training mode is: {config.model.mode}")
    return GaussianDiffusion(
        config=config,
        # β: variance term
        betas=betas,
        # Determine the target of the final prediction of the model, 
        # EPSILON is the noise in the reconstruction method, and START_X is the original x_0
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not config.predict_xstart else gd.ModelMeanType.START_X
        ),
        # Determine the type of model variance, 
        # in general, the variance is fixed, no trainable variance is needed, 
        # if trainable, the final output dimension of the model should be doubled
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE  # choose
                if not config.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not config.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
    )
