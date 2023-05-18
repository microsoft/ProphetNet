from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(config, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if config.schedule_sampler == "uniform":
        return UniformSampler(diffusion)
    elif config.schedule_sampler == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    elif config.schedule_sampler == "xy_uniform":
        return XYUniformSampler(config, diffusion)
    elif config.schedule_sampler == "fixed_xy":
        return FixedUniformSampler(diffusion)
    elif config.schedule_sampler == "xy3_uniform":
        return XY3UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {config.schedule_sampler}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device, step_ratio=None):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights
    
    
class FixedUniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

    def sample(self, batch_size, device, step_ratio, seq_len=None, *args, **kwargs):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        example:
        num_timesteps = 3, seq_len = 5
        2 3 4 5 6
        1
        0         (0, 7[num_time_steps+seq_len-1])
        """

        w = np.ones([self.diffusion.num_timesteps + seq_len])
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)

        # TODO
        middle_point = th.stack([
            th.clamp((seq_len - 1) - indices, 0),
            th.clamp(indices - (seq_len - 1), 0)
        ], dim=-1)
        middle_point_x, middle_point_y = middle_point[:, 0], middle_point[:, 1]
        
        # Fix the (2l, t) and (l ,0) => (t - 0) / (2l - l) = t / l
        xs = th.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).type_as(middle_point)
        bias = middle_point_y.float() - middle_point_x.float().mul(seq_len - 1).div(self.diffusion.num_timesteps - 1)
        ys = xs.float().mul(seq_len - 1).div(self.diffusion.num_timesteps - 1) + bias.unsqueeze(-1)
        ys = ys.round().clamp(0, self.diffusion.num_timesteps - 1).long()
        
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        weights = weights.unsqueeze(-1).repeat(1, seq_len)
        return ys, weights
    

class XYUniformSampler(ScheduleSampler):
    def __init__(self, config, diffusion, history_per_term=10, uniform_prob=0.001):
        self.config = config
        self.diffusion = diffusion
        self.scale = getattr(self.config, "end_point_scale", 2.0)

        if config.loss_aware:
            self.history_per_term = history_per_term
            self.uniform_prob = uniform_prob
            self._loss_history = np.zeros(
                [diffusion.num_timesteps, history_per_term], dtype=np.float64
            )
            self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps + self.config.tgt_len], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights
    
    def update_with_local_losses(self, local_ts, local_losses):
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs, self.config.tgt_len).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs, self.config.tgt_len).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.float().max().round().long().item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.float().max().item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)  # List -> len=bs*world_size, element=tensor[tgt_len]
    
    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

        # TODO: too slow
        # for t, loss in zip(ts, losses):
        #     assert len(t) == len(loss) == self.config.tgt_len
        #     for t_j, (t_i, loss_item) in enumerate(zip(t, loss)):
        #         t_i, loss_item = t_i.item(), loss_item.item()
        #         if self._loss_counts[t_i][t_j] == self.history_per_term:  # [timesteps, tgt_len]
        #             # Shift out the oldest loss term.
        #             self._loss_history[t_i, t_j, :-1] = self._loss_history[t_i, t_j, 1:]
        #             self._loss_history[t_i, t_j, -1] = loss_item
        #         else:
        #             self._loss_history[t_i, t_j, self._loss_counts[t_i][t_j]] = loss_item
        #             self._loss_counts[t_i][t_j] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

    def sample(self, batch_size, device, seq_len, step_ratio=None):
        """
        example:
        num_timesteps = 3, seq_len = 5
        2 3 4 5 6
        1
        0         (0, 7[num_time_steps+seq_len-1])
        """
        
        # time_start = 0
        # time_end = self.diffusion.num_timesteps + seq_len
        # if step_ratio >= float(self.config.ratio_thre):
        #     time_start = seq_len

        # w = np.ones([time_end - time_start] )
        
        if not self.config.pred_len:
            assert (seq_len == self.config.tgt_len).all()

        if self.config.loss_aware:
            w = self.weights()
        else:
            w = np.ones([max(seq_len) + self.diffusion.num_timesteps])

        p = w / np.sum(w)
        # indices_np = np.random.choice(np.arange(time_start, time_end), size=(batch_size,), p=p)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)  # [bs, ]->(0, time_step + seq_len)
        indices = th.from_numpy(indices_np).long().to(device)

        middle_point = th.stack([
            th.clamp((seq_len - 1) - indices, 0),
            th.clamp(indices - (seq_len - 1), 0)
        ], dim=-1)

        if self.config.pred_len:
            end_point_x = (self.scale * seq_len).type_as(middle_point)
            end_point_y = th.tensor(self.diffusion.num_timesteps - 1).repeat(batch_size).type_as(middle_point)
            end_point = th.stack([end_point_x, end_point_y], dim=-1)
        else:
            end_point = th.tensor(
                [[int(self.scale * max(seq_len)), self.diffusion.num_timesteps - 1]]
            ).repeat(batch_size, 1).type_as(middle_point)

        # the part of padding will be mask
        xs = th.arange(max(seq_len)).unsqueeze(0).repeat(batch_size, 1).type_as(middle_point)
        """
        (y - end_y) / (middle_y - end_y) = (x - end_x) / (middle_x - end_x)
        => y = (x - end_x) / (middle_x - end_x) * (middle_y - end_y) + end_y
        """
        end_point = end_point.unsqueeze(-1)
        middle_point = middle_point.unsqueeze(-1)
        ys = (xs.float() - end_point[:, 0].float()
              ).div(middle_point[:, 0].float() - end_point[:, 0].float()
                    ).mul(middle_point[:, 1].float() - end_point[:, 1].float()
                          ).add(end_point[:, 1].float())
        ys = ys.round().clamp(0, self.diffusion.num_timesteps - 1).long().to(device)
        
        # weights_np = 1 / ((time_end - time_start) * p[indices_np - time_start])
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        weights = weights.unsqueeze(-1).repeat(1, max(seq_len))
        return ys, weights


class XY3UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self.config = diffusion.config
        self.scale = getattr(self.config, "end_point_scale", 1.0)
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

    def sample(self, batch_size, device, seq_len=None, step_ratio=None, *args, **kwargs):

        w = np.ones([self.diffusion.num_timesteps + seq_len])
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)

        middle_point = th.stack([
            th.clamp(seq_len - 1 - indices, 0),
            th.clamp(indices - seq_len + 1, 0)
        ], dim=-1).float()

        middle_point_x = middle_point[:, 0]
        middle_point_y = middle_point[:, 1]
        end_point_x = th.tensor(
            [self.scale * seq_len]).repeat(batch_size).type_as(middle_point)
        end_point_y = th.rand(batch_size).type_as(middle_point).mul(self.diffusion.num_timesteps).add(middle_point_y)
        end_point = th.stack([end_point_x, end_point_y], dim=-1)

        xs = th.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).type_as(middle_point)
        """
        (y-y2)/(y1-y2) = (x-x2)/(x1-x2)
        """
        end_point = end_point.unsqueeze(-1)
        middle_point = middle_point.unsqueeze(-1)
        ys = (xs.float() - end_point[:, 0].float()).div(middle_point[:, 0].float() - end_point[:, 0].float()). \
            mul(middle_point[:, 1].float() - end_point[:, 1].float()).add(end_point[:, 1].float())
        ys = ys.round().clamp(0, self.diffusion.num_timesteps - 1).long()

        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        weights = weights.unsqueeze(-1).repeat(1, seq_len)

        return ys, weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
