from model.Diffusion_LM import Diffusion_LM, CrossAttention_Diffusion_LM
from diffusion_util import gaussian_diffusion as gd
from diffusion_util.respace import SpacedDiffusion, space_timesteps
from diffusion_util.gaussian_diffusion import GaussianDiffusion



def create_model_and_diffusion(
    args
):
    model = create_model(
        model_channels=args.model_channels,
        learn_sigma=args.learn_sigma,
        dropout=args.dropout,
        model_arch=args.model_arch,
        in_channel=args.in_channel,
        out_channel=args.out_channel,
        vocab_size=args.vocab_size,
        config_name=args.config_name,
        logits_mode=args.logits_mode,
        init_pretrained=args.init_pretrained,
        token_emb_type=args.token_emb_type,
    )
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        sigma_small=args.sigma_small,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        model_arch=args.model_arch,
        training_mode=args.training_mode,
    )
    return model, diffusion

'''
create diffusion model
'''
def create_model(
    model_channels,
    learn_sigma,
    dropout,
    model_arch,
    in_channel=8,
    out_channel=8,
    vocab_size=None,
    config_name='',
    logits_mode=1,
    init_pretrained=True,
    token_emb_type='pretrain',
):
    print(f'creating model, based on {model_arch}')

    if model_arch == 'transformer':

        return Diffusion_LM(
            in_channels=in_channel,
            model_channels=model_channels,
            out_channels=(out_channel if not learn_sigma else out_channel*2),
            dropout=dropout,
            config_name=config_name,
            vocab_size=vocab_size,
            logits_mode=logits_mode,
            init_pretrained=init_pretrained,
            token_emb_type=token_emb_type,
        )

    elif model_arch == 's2s_CAT':

        return CrossAttention_Diffusion_LM(
            in_channels=in_channel,
            model_channels=model_channels,
            out_channels=(out_channel if not learn_sigma else out_channel * 2),
            dropout=dropout,
            config_name=config_name,
            vocab_size=vocab_size,
            logits_mode=logits_mode,
            init_pretrained=init_pretrained,
            token_emb_type=token_emb_type,
        )
    else:
        raise NotImplementedError


'''
create diffusion process
'''
def create_gaussian_diffusion(
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="cosine",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    model_arch='transformer',
    training_mode='e2e',
):

    # β , Determine according to the maximum T and variance schedule
    print("noise_schedule: ", noise_schedule)
    print("Diffusion Steps: ", steps)
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    print("betas: ", betas)

    # determine the loss function used in training
    if training_mode == 'e2e' or training_mode == 's2s':
        # end to end training
        if use_kl:
            loss_type = gd.LossType.E2E_KL
        else:
            loss_type = gd.LossType.E2E_MSE
    elif training_mode == 'e2e-simple':
        if use_kl:
            loss_type = gd.LossType.E2E_Simple_KL
        else:
            loss_type = gd.LossType.E2E_Simple_MSE

    else:
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE


    if not timestep_respacing:
        timestep_respacing = [steps]
    print("Diffusion Loss Type: ", loss_type, "  , Whether to learn sigma: ", learn_sigma)
    print("Diffusion predict xstart: ", predict_xstart)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        model_arch=model_arch,
        training_mode=training_mode,
    )


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}