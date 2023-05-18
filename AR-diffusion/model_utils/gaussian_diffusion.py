"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from random import random


def timestep_embedding(config, timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0,
                                             end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    if config.schedule_sampler == 'uniform':
        args = timesteps[:, None].float() * freqs[None]
    else:
        args = timesteps[:, :, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  # scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        """
        \bar{α}_t = 1 - \sqrt{t/T + s} (sqrt schedule)
        (s is a small constant that corresponds to the starting noise level)
        β_t = 1 - α_t = 1 - \bar{α}_t / \bar{α}_{t-1}
        x_t = \sqrt{1-β_t} * x_{t-1}
        """
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(
        self,
        config,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
    ):
        self.config = config
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = config.rescale_timesteps
        self.training_mode = config.model.mode

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        # cumprod，Multiplication function
        # each dimension multiplies all the numbers in the previous dimension, and calculates \bar{α}_{t}
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # \bar{α}_{t-1} \bar{α}_{t+1}
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # \sqrt(\bar{α}_{t})
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        # \sqrt(1 - \bar{α}_{t})
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        # \log(1 - \bar{α}_{t})
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        # \sqrt(1 / \bar{α}_{t})
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        # \sqrt(1 / \bar{α}_{t} - 1)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # the variance coefficient of q(x_{t-1} | x_t, x_0) is β_{t} * [(1 - \bar{α}_{t-1}) / (1 - \bar{α}_{t})]
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # the coefficient of x_0 in q(x_{t-1} | x_t, x_0)
        # β_{t} * sqrt(\bar{α}_{t-1}) / (1 - \bar{α}_{t})
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # the coefficient of x_t in q(x_{t-1} | x_t, x_0)
        # (1-\bar{α}_{t-1}) * sqrt(α) / (1 - \bar{α}_{t})
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def training_losses(self, model, input_text, t, is_dev=False):
        if self.training_mode == 's2s':  # choose
            return self.training_losses_s2s(model, input_text, t, is_dev=is_dev)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod,
                                 t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(
            1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        # x_t = \sqrt{\bar{α}_t} * x_0 + (1 - \sqrt{\bar{α}_t}) * noise
        
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1,
                                 t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2,
                                   t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, self_cond=None, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [bs x seq_len x embed_dim] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.size(0), x.size(-1)
        if self.config.schedule_sampler == 'uniform':
            assert t.shape == (B,)

        model_output, _ = model(x, self._scale_timesteps(t), x_self_cond=self_cond, **model_kwargs)

        # Learning Variance
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, x.size(1), C * 2)
            model_output, model_var_values = th.split(
                model_output, C, dim=-1)

            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                # Select the nearest embedding by KNN, that is predicting the x_0
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(
                1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        pred_lengs=None,
        top_p=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (bs, seq_len, emb_dim).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        if self.config.schedule_sampler == 'xy_uniform':
            if self.config.skip_sample:
                print('**************skip xy_uniform sample**************')
                for sample in self.p_skip_xy_sample_loop_progressive(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    pred_lengs=pred_lengs,
                    top_p=top_p,
                ):
                    final = sample
            else:
                print('**************xy_uniform sample**************')
                for sample in self.p_xy_sample_loop_progressive(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    top_p=top_p,
                ):
                    final = sample
        elif self.config.schedule_sampler == 'fixed_xy':
            if self.config.skip_sample:
                print('**************skip fixed xy_uniform sample**************')
                for sample in self.p_skip_fixedxy_sample_loop_progressive(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    top_p=top_p,
                ):
                    final = sample
        else:
            if self.config.skip_sample:
                print('**************skip sample**************')
                for sample in self.p_skip_sample_loop_progressive(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    top_p=top_p,
                ):
                    final = sample
            else:
                print('**************standard sample**************')
                for sample in self.p_sample_loop_progressive(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    top_p=top_p,
                ):
                    final = sample

        return final["sample"]
    
    # **************** standard ****************
    def p_sample(
        self, 
        model, 
        x, 
        t, 
        self_cond=None, 
        clip_denoised=False, 
        denoised_fn=None, 
        model_kwargs=None,
        top_p=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # out: x_{t-1}
        out = self.p_mean_variance(
            model,
            x,
            t,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # t: (bs, 1, 1), the value is 0/1
        )  # no noise when t == 0
        if self.config.schedule_sampler != 'uniform':
            nonzero_mask = nonzero_mask.reshape(noise.size(0), noise.size(1), 1)
        # \exp^{0.5 * \log σ^2} = σ, i.e. standard derivation
        # out["log_variance"], noise: (bs, seq_len, embed_dim)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"],
            'greedy_mean': out["mean"], 
            'out': out
        }

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            text = noise
        else:
            text = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]  # reverse the list

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # The time step traverses sequentially from back to front
        x_start = None
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                self_cond = x_start if self.config.infer_self_condition else None
                out = self.p_sample(
                    model,
                    text,
                    t,
                    self_cond=self_cond,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                text = out["sample"]
                x_start = out["pred_xstart"]
                
    def p_xy_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            text = noise
        else:
            text = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps + shape[1]))[::-1]

        scale = getattr(self.config, "end_point_scale", 1.0)
        end_point = th.tensor(
            [[int(scale * shape[1]), self.num_timesteps - 1]]
        ).repeat(shape[0], 1).to(device)
        xs = th.arange(shape[1]).unsqueeze(0).repeat(shape[0], 1).to(device)
        end_point = end_point.unsqueeze(-1)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # The time step traverses sequentially from back to front
        x_start = None
        for i in indices:
            i = th.tensor([i] * shape[0], device=device)

            middle_point = th.stack([
                th.clamp((shape[1] - 1) - i, 0),
                th.clamp(i - (shape[1] - 1), 0)
            ], dim=-1)
            middle_point = middle_point.unsqueeze(-1)

            t = (xs.float() - end_point[:, 0].float()
                 ).div(middle_point[:, 0].float() - end_point[:, 0].float()
                       ).mul(middle_point[:, 1].float() - end_point[:, 1].float()
                             ).add(end_point[:, 1].float())
            t = t.round().clamp(0, self.num_timesteps - 1).long()

            with th.no_grad():
                self_cond = x_start if self.config.infer_self_condition else None
                out = self.p_sample(
                    model,
                    text,
                    t,
                    self_cond=self_cond,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                text = out["sample"]
                x_start = out["pred_xstart"]

    def p_sample_loop_langevin_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        langevin_func=None,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                if langevin_func is not None:
                    out['t'] = t
                    out['img'] = img
                    out = langevin_func(out)
                yield out
                img = out["sample"]

    # **************** skip ****************
    def p_skip_sample(
        self,
        model,
        x,
        t,
        nt,
        self_cond=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        top_p=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # out: x_{t-1}
        out = self.p_mean_variance(
            model,
            x,
            t,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        alpha_nt = _extract_into_tensor(self.alphas_cumprod, nt, x.shape)
        alpha_t = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        t_div_nt = alpha_t / alpha_nt
        one_minus_nt_div_t = (1 - alpha_nt) / (1 - alpha_t)

        out["mean"] = t_div_nt.sqrt() * one_minus_nt_div_t * x + \
            alpha_nt.sqrt() * (1 - alpha_t / alpha_nt) / (1 - alpha_t) * out["pred_xstart"]
        out["variance"] = (1 - t_div_nt) * one_minus_nt_div_t
        out["log_variance"] = out["variance"].log()

        if top_p is not None and top_p > 0:
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)
        nonzero_mask = (
            (nt != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # t: (bs, 1, 1), the value is 0/1
        )  # no noise when t == 0
        # \exp^{0.5 * \log σ^2} = σ, i.e. standard derivation
        # out["log_variance"], noise: (bs, seq_len, embed_dim)
        if nonzero_mask.size(0) != noise.size(0):
            nonzero_mask = nonzero_mask.reshape(noise.size(0), noise.size(1), 1)
            
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            'greedy_mean': out["mean"],
            'out': out
        }

    def p_skip_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            text = noise
        else:
            text = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1][::self.config.skip_timestep]
        if indices[-1] != 0:
            indices.append(0)
        src_indices = indices[:-1]
        tgt_indices = indices[1:]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            src_indices = tqdm(src_indices)

        # The time step traverses sequentially from back to front
        x_start = None
        for src_i, tgt_i in zip(src_indices, tgt_indices):
            src_i = th.tensor([src_i] * shape[0], device=device)
            tgt_i = th.tensor([tgt_i] * shape[0], device=device)
            with th.no_grad():
                self_cond = x_start if self.config.infer_self_condition else None
                out = self.p_skip_sample(
                    model,
                    text,
                    src_i,
                    tgt_i,
                    self_cond=self_cond,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                text = out["sample"]
                x_start = out["pred_xstart"]
                
    def p_skip_xy_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        pred_lengs=None,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            text = noise
        else:
            text = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps + shape[1]))[::-1]
        skip_timestep = (self.num_timesteps + shape[1]) // self.config.gen_timesteps
        
        if self.config.skip_sample:
            indices = indices[::skip_timestep][:self.config.gen_timesteps]
            assert len(indices) == self.config.gen_timesteps
        if indices[-1] != 0:
            indices.append(0)
        src_indices = indices[:-1]
        tgt_indices = indices[1:]

        scale, batch_size = self.config.end_point_scale, shape[0]
        if self.config.pred_len:
            end_point_x = 2.0 * pred_lengs
            end_point_y = th.tensor(self.num_timesteps - 1).repeat(batch_size).to(device)
            end_point = th.stack([end_point_x, end_point_y], dim=-1)
        else:
            end_point = th.tensor(
                [[int(scale * shape[1]), self.num_timesteps - 1]]
            ).repeat(batch_size, 1).to(device)

        xs = th.arange(shape[1]).unsqueeze(0).repeat(batch_size, 1).to(device)
        end_point = end_point.unsqueeze(-1)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            src_indices = tqdm(src_indices)

        # The time step traverses sequentially from back to front
        x_start = None
        # idx, wp_list = 0, []
        # for i in range(self.config.batch_size):
        #     wp_list.append(open(f'/wutong/DiffusionXY/case_out/out{self.config.gen_timesteps}_{i}.txt', 'w'))
        for src_i, tgt_i in zip(src_indices, tgt_indices):
            src_i = th.tensor([src_i] * batch_size, device=device)
            if self.config.pred_len:
                middle_point = th.stack([
                    th.clamp((pred_lengs - 1) - src_i, 0),
                    th.clamp(src_i - (pred_lengs - 1), 0)
                ], dim=-1)
            else:
                middle_point = th.stack([
                    th.clamp((shape[1] - 1) - src_i, 0),
                    th.clamp(src_i - (shape[1] - 1), 0)
                ], dim=-1)
            middle_point = middle_point.unsqueeze(-1)
            src_t = (xs.float() - end_point[:, 0].float()
                 ).div(middle_point[:, 0].float() - end_point[:, 0].float()
                       ).mul(middle_point[:, 1].float() - end_point[:, 1].float()
                             ).add(end_point[:, 1].float())
            src_t = src_t.round().clamp(0, self.num_timesteps - 1).long()

            tgt_i = th.tensor([tgt_i] * batch_size, device=device)
            if self.config.pred_len:
                middle_point = th.stack([
                    th.clamp((pred_lengs - 1) - tgt_i, 0),
                    th.clamp(tgt_i - (pred_lengs - 1), 0)
                ], dim=-1)
            else:
                middle_point = th.stack([
                    th.clamp((shape[1] - 1) - tgt_i, 0),
                    th.clamp(tgt_i - (shape[1] - 1), 0)
                ], dim=-1)
            middle_point = middle_point.unsqueeze(-1)
            tgt_t = (xs.float() - end_point[:, 0].float()
                 ).div(middle_point[:, 0].float() - end_point[:, 0].float()
                       ).mul(middle_point[:, 1].float() - end_point[:, 1].float()
                             ).add(end_point[:, 1].float())
            tgt_t = tgt_t.round().clamp(0, self.num_timesteps - 1).long()

            with th.no_grad():
                self_cond = x_start if self.config.infer_self_condition else None
                out = self.p_skip_sample(
                    model,
                    text,
                    src_t,
                    tgt_t,
                    self_cond=self_cond,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                text = out["sample"]
                x_start = out["pred_xstart"]

        #     idx += 1
        #     if idx > 950:
        #         from transformers import AutoTokenizer
        #         tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)

        #         for i in range(self.config.batch_size):
        #             if th.distributed.get_rank() == 0:
        #                 wp_list[i].write('*********Step: ' + str(idx) + '************\n')
                    
        #             logits = model.module.get_logits(text[i])  # (bs, seq_len, vocab_size)
        #             sample_id = th.topk(logits, k=5, dim=-1)

        #             true_text = tokenizer.batch_decode(sample_id.indices, skip_special_tokens=False)
        #             if th.distributed.get_rank() == 0:
        #                 wp_list[i].write(str(true_text) + '\n')
        #                 wp_list[i].write(str(sample_id.values) + '\n')
        # exit(0)    
                
    def p_skip_fixedxy_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            text = noise
        else:
            text = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps + shape[1]))[::-1]
        if self.config.skip_sample:
            indices = indices[::self.config.skip_timestep]
        if indices[-1] != 0:
            indices.append(0)
        # indices = indices[::-1]  # reverse the list
        src_indices = indices[:-1]
        tgt_indices = indices[1:]

        xs = th.arange(shape[1]).unsqueeze(0).repeat(shape[0], 1).to(device)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            src_indices = tqdm(src_indices)

        # The time step traverses sequentially from back to front
        x_start = None
        for src_i, tgt_i in zip(src_indices, tgt_indices):
            src_i = th.tensor([src_i] * shape[0], device=device)
            middle_point = th.stack([
                th.clamp((shape[1] - 1) - src_i, 0),
                th.clamp(src_i - (shape[1] - 1), 0)
            ], dim=-1)
            middle_point_x, middle_point_y = middle_point[:, 0], middle_point[:, 1]
            # Fix the (2l, t) and (l ,0) => (t - 0) / (2l - l) = t / l
            bias = middle_point_y.float() - middle_point_x.float().mul(shape[1] - 1).div(self.num_timesteps - 1)
            src_t = xs.float().mul(shape[1] - 1).div(self.num_timesteps - 1) + bias.unsqueeze(-1)
            src_t = src_t.round().clamp(0, self.num_timesteps - 1).long()

            tgt_i = th.tensor([tgt_i] * shape[0], device=device)
            middle_point = th.stack([
                th.clamp((shape[1] - 1) - tgt_i, 0),
                th.clamp(tgt_i - (shape[1] - 1), 0)
            ], dim=-1)
            middle_point_x, middle_point_y = middle_point[:, 0], middle_point[:, 1]
            # Fix the (2l, t) and (l ,0) => (t - 0) / (2l - l) = t / l
            bias = middle_point_y.float() - middle_point_x.float().mul(shape[1] - 1).div(self.num_timesteps - 1)
            tgt_t = xs.float().mul(shape[1] - 1).div(self.num_timesteps - 1) + bias.unsqueeze(-1)
            tgt_t = tgt_t.round().clamp(0, self.num_timesteps - 1).long()

            with th.no_grad():
                self_cond = x_start if self.config.infer_self_condition else None
                out = self.p_skip_sample(
                    model,
                    text,
                    src_t,
                    tgt_t,
                    self_cond=self_cond,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                text = out["sample"]
                x_start = out["pred_xstart"]

    # **************** ddim ****************
    def ddim_sample(
        self,
        model,
        x,
        t,
        next_t,
        self_cond=None, 
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            self_cond=self_cond, 
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, next_t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if self.config.schedule_sampler != 'uniform':
            nonzero_mask = nonzero_mask.reshape(noise.size(0), noise.size(1), 1)
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            sample = langevin_fn(sample, mean_pred, sigma,
                                 self.alphas_cumprod_prev[t[0]], t, x)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(
                self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(
            self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        pred_lengs=None,
        eta=0.0,
        top_p=-1.0,
        langevin_fn=None,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        if self.config.schedule_sampler == 'xy_uniform':
            print('**************xy_uniform ddim sample**************')
            for sample in self.ddim_sample_xy_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                langevin_fn=langevin_fn,
            ):
                final = sample
        else:
            print('**************standard ddim sample**************')
            for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                langevin_fn=langevin_fn,
            ):
                final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            text = noise
        else:
            text = th.randn(*shape, device=device)

        indices = list(range(0, self.num_timesteps, self.config.skip_timestep))[::-1]
        indices_next = indices[1:] + [0]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)

        for i, j in zip(indices, indices_next):
            t = th.tensor([i] * shape[0], device=device)
            next_t = th.tensor([j] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    text,
                    t,
                    next_t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    langevin_fn=langevin_fn,
                )
                yield out
                text = out["sample"]
            
    def ddim_sample_xy_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            text = noise
        else:
            text = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps + shape[1]))[::-1]
        skip_timestep = (self.num_timesteps + shape[1]) // self.config.gen_timesteps
        
        if self.config.ddim_sample:
            indices = indices[::skip_timestep][:self.config.gen_timesteps]
            assert len(indices) == self.config.gen_timesteps

        if indices[-1] != 0:
            indices.append(0)
        
        src_indices = indices[:-1]
        tgt_indices = indices[1:]

        end_point = th.tensor(
            [[int(2.0 * shape[1]), self.num_timesteps - 1]]
        ).repeat(shape[0], 1).to(device)
        xs = th.arange(shape[1]).unsqueeze(0).repeat(shape[0], 1).to(device)
        end_point = end_point.unsqueeze(-1)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            src_indices = tqdm(src_indices)

        # The time step traverses sequentially from back to front
        x_start = None
        for src_i, tgt_i in zip(src_indices, tgt_indices):
            src_i = th.tensor([src_i] * shape[0], device=device)
            middle_point = th.stack([
                th.clamp((shape[1] - 1) - src_i, 0),
                th.clamp(src_i - (shape[1] - 1), 0)
            ], dim=-1)
            middle_point = middle_point.unsqueeze(-1)
            src_t = (xs.float() - end_point[:, 0].float()
                 ).div(middle_point[:, 0].float() - end_point[:, 0].float()
                       ).mul(middle_point[:, 1].float() - end_point[:, 1].float()
                             ).add(end_point[:, 1].float())
            src_t = src_t.round().clamp(0, self.num_timesteps - 1).long()

            tgt_i = th.tensor([tgt_i] * shape[0], device=device)
            middle_point = th.stack([
                th.clamp((shape[1] - 1) - tgt_i, 0),
                th.clamp(tgt_i - (shape[1] - 1), 0)
            ], dim=-1)
            middle_point = middle_point.unsqueeze(-1)
            tgt_t = (xs.float() - end_point[:, 0].float()
                 ).div(middle_point[:, 0].float() - end_point[:, 0].float()
                       ).mul(middle_point[:, 1].float() - end_point[:, 1].float()
                             ).add(end_point[:, 1].float())
            tgt_t = tgt_t.round().clamp(0, self.num_timesteps - 1).long()

            with th.no_grad():
                self_cond = x_start if self.config.infer_self_condition else None
                out = self.ddim_sample(
                    model,
                    text,
                    src_t,
                    tgt_t,
                    self_cond=self_cond,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    langevin_fn=langevin_fn,
                )
                yield out
                text = out["sample"]
                x_start = out["pred_xstart"]

    def get_x_start(self, x_start_mean, std):
        '''
        Using the interpolating policy OR using the convolution policy...
        :param x_start_mean:
        :return:
        '''
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        
        return (
            x_start_mean + std * noise
        )

    def token_discrete_loss(self, x_0, get_logits, input_ids):
        reshaped_x_0 = x_0
        logits = get_logits(reshaped_x_0)  # (bsz, seqlen, vocab)
        if self.config.pred_len:
            loss_fct = th.nn.CrossEntropyLoss(reduction='none', ignore_index=self.config.pad_value)
        else:
            loss_fct = th.nn.CrossEntropyLoss(reduction='none', label_smoothing=self.config.label_smooth)
        decoder_nll = loss_fct(
            logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        return decoder_nll

    def x0_helper(self, model_output, x, t):
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(
                x_t=x, t=t, xprev=model_output)
            pred_prev = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:  # choose
                pred_xstart = model_output
            else:
                pred_xstart = self._predict_xstart_from_eps(
                    x_t=x, t=t, eps=model_output)
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:
            raise NotImplementedError(self.model_mean_type)
        return {'pred_xprev': pred_prev, 'pred_xstart': pred_xstart}

    def model_predictions(self, x, t, prev_output):
        if not self.config.predict_xstart:
            pred_noise = prev_output
            x_start = self._predict_xstart_from_eps(x, t, prev_output)

        elif self.config.predict_xstart:
            pred_noise = self._predict_eps_from_xstart(x, t, prev_output)
            x_start = prev_output

        return {'pred_noise': pred_noise, 'pred_x_start': x_start}

    def training_losses_s2s(self, model, input_text, t, is_dev=False):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # for s2s (p: src, q (x_0_mean): tgt)
        q_input_ids = input_text['tgt_input_ids'].long().to(t.device)
        x_start_mean = model.module.get_embeds(q_input_ids)  # because of DDP
        p_input_ids = input_text['src_input_ids'].long().to(t.device)
        p_attention_mask = input_text['src_attention_mask'].long().to(t.device)
        q_attention_mask = input_text['tgt_attention_mask'].long().to(t.device) if self.config.pred_len else None
        tgt_length = input_text['length'].long().to(t.device) if self.config.pred_len else None

        # the variance of x_0 is \sqrt{(1 - \bar{α}_t)} when t=0
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),  # t=0
                                   x_start_mean.shape)
        x_start_log_var = 2 * th.log(std)
        x_start = self.get_x_start(x_start_mean, std)  # (bs, seq_len, hz)

        noise = None
        if noise is None:
            noise = th.randn_like(x_start)
        # reparametrization trick.
        x_t = self.q_sample(x_start, t, noise=noise)
        get_logits = model.module.get_logits  # passed in is the method

        terms = {}

        if self.loss_type == LossType.E2E_MSE or self.loss_type == LossType.RESCALED_MSE:
            x_self_cond = None
            if self.config.self_condition and random() < 0.5:
                with th.no_grad():
                    prev_output, length_out = model(tgt_emb=x_t, 
                                                    timesteps=self._scale_timesteps(t),
                                                    x_self_cond=x_self_cond,
                                                    src_input_ids=p_input_ids,
                                                    src_attention_mask=p_attention_mask,
                                                    tgt_attention_mask=q_attention_mask,
                                                    tgt_length=tgt_length,)
                    x_self_cond = self.model_predictions(x_t, t, prev_output)['pred_x_start']
                    # beacause of the DDP, the detach_() is unavailable
                    # detach and detach_ are all stop gradient, but detach will generate a new tensor
                    x_self_cond.detach()

            # model: LM model, input the src, output the tgt
            model_output, length_out = model(tgt_emb=x_t, 
                                             timesteps=self._scale_timesteps(t),
                                             x_self_cond=x_self_cond,
                                             src_input_ids=p_input_ids,
                                             src_attention_mask=p_attention_mask,
                                             tgt_attention_mask=q_attention_mask,
                                             tgt_length=tgt_length,)

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,  # choose
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            # calculate the MSE loss about \bar{x_0}(contain variance)
            terms["mse"] = ((target - model_output) ** 2).mean(-1)  # [bs, seqlen]
            if self.config.schedule_sampler == 'uniform':
                terms["mse"] = terms["mse"].mean(-1)  # [bs]

            # only when t=0, its distribution is completely close to the distribution of embedding i.e. t0_loss, 
            # calculate the MSE loss about x_0(no variance)
            model_out_x_start = self.x0_helper(model_output, x_t, t)['pred_xstart']
            t0_mask = (t == 0)
            t0_loss = ((x_start_mean - model_out_x_start) ** 2).mean(-1)  # [bs, seqlen]
            if self.config.schedule_sampler == 'uniform':
                t0_loss = t0_loss.mean(-1)  # [bs]

            # only when t=0, its distribution is completely close to the distribution of embedding i.e. t0_loss, 
            # otherwise it is close to x_{0} i.e. terms["mse"]
            terms["t0_loss"] = t0_loss
            terms["mse_pre"] = terms["mse"]  # [bs, seqlen] / [bs]
            terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])  # [bs, seqlen] / [bs]

            # let the x_T approximate the Guassian, \mu = 0
            if self.config.predict_x_start:
                x_output = model_out_x_start
            else:
                x_output = x_start

            out_mean, _, _ = self.q_mean_variance(x_output, th.LongTensor(
                [self.num_timesteps - 1]).to(x_output.device))
            # tT_loss = mean_flat(out_mean ** 2)
            tT_loss = (out_mean ** 2).mean(-1)
            if self.config.schedule_sampler == 'uniform':
                tT_loss = tT_loss.mean(-1)
            terms["tT_loss"] = tT_loss

            # At each step, the cross-entropy with the real data is calculated.
            decoder_nll = self.token_discrete_loss(x_output, get_logits, q_input_ids)
            terms["decoder_nll"] = decoder_nll
            if self.config.schedule_sampler == 'uniform':
                terms["decoder_nll"] = terms["decoder_nll"].mean(-1)
                
            loss_fct = th.nn.CrossEntropyLoss(reduction='none')
            if self.config.pred_len:
                # length_out: [bs, seq_len] / tgt_length: [bs,]
                tgt_length = tgt_length.view(-1)
                # reduce the loss scale
                terms["length_loss"] = self.config.length_factor * loss_fct(
                    length_out, tgt_length).unsqueeze(-1)
                if is_dev:
                    top1_len = th.topk(length_out, 1, dim=-1)[1]
                    correct = th.eq(top1_len.view(-1), tgt_length.view(-1)).sum(0)
                    terms["top-1_acc"] = correct / length_out.size(1)

                    top5_len = th.topk(length_out, 5, dim=-1)[1]
                    correct = th.eq(
                        top5_len.view(-1), 
                        tgt_length.view(-1, 1).expand_as(top5_len).contiguous().view(-1)
                    ).sum(0)
                    terms["top-5_acc"] = correct / length_out.size(1)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                if self.config.pred_len:
                    terms["loss"] = terms["mse"] + (terms["decoder_nll"] + terms["tT_loss"]) + terms["length_loss"]
                else:
                    terms["loss"] = terms["mse"] + (terms["decoder_nll"] + terms["tT_loss"])
                # terms["loss"] = terms["mse"] + (1.0/self.num_timesteps) * decoder_nll + \
                #                 (1.0/self.num_timesteps) * tT_loss
        else:
            raise NotImplementedError(self.loss_type)

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
