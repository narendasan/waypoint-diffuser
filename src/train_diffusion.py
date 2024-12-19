import torch
import numpy as np
import collections
from tqdm import tqdm

import gymnasium as gym
import gym_pusht
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_network import ConditionalUnet1D
from push_t_dataset import PushTStateDataset, normalize_data, unnormalize_data
from train_waypoint import WaypointUtil

class NormalDiffusion():
    def __init__(self, dataset_path,
                 pred_horizon=16, obs_horizon=2, action_horizon=8,
                 batch_size=256, num_diffusion_iters=100):
        # parameters
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.batch_size = batch_size
        self.num_diffusion_iters = num_diffusion_iters

        # dataset and data loader
        self.dataset_path = dataset_path
        self.dataset = PushTStateDataset(dataset_path, pred_horizon, obs_horizon, action_horizon)

        # print data stats
        print("obs.shape:", self.dataset[0]['obs'].shape)
        print("action.shape", self.dataset[0]['action'].shape)
        # save data stats
        self.obs_dim = self.dataset[0]['obs'].shape[-1]
        self.action_dim = self.dataset[0]['action'].shape[-1]

        # create network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * obs_horizon
        )

        # create diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise
            prediction_type='epsilon'
        )

    def to_device(self, device):
        self.device = torch.device(device)
        _ = self.noise_pred_net.to(self.device)

    def train(self, num_epochs=100):
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process after each epoch
            persistent_workers=True)

        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        ema = EMAModel(
            parameters=self.noise_pred_net.parameters(),
            power=0.75)

        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(),
            lr=1e-4, weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * num_epochs
        )

        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = []
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        nobs = nbatch['obs'].to(self.device)
                        naction = nbatch['action'].to(self.device)
                        B = nobs.shape[0]

                        # observation as FiLM conditioning
                        # (B, obs_horizon, obs_dim)
                        obs_cond = nobs[:,:self.obs_horizon,:]
                        # (B, obs_horizon * obs_dim)
                        obs_cond = obs_cond.flatten(start_dim=1)

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=self.device
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = self.noise_scheduler.add_noise(
                            naction, noise, timesteps)

                        # predict the noise residual
                        noise_pred = self.noise_pred_net(
                            noisy_actions, timesteps, global_cond=obs_cond)

                        # L2 loss
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        ema.step(self.noise_pred_net.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))

        # Weights of the EMA model
        # is used for inference
        self.ema_noise_pred_net = self.noise_pred_net
        ema.copy_to(self.ema_noise_pred_net.parameters())


    def load_pretrain(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.ema_noise_pred_net = self.noise_pred_net
        self.ema_noise_pred_net.load_state_dict(state_dict)

    def eval(self, start_state=None):
        # limit enviornment interaction to 200 steps before termination
        max_steps = 200
        env = gym.make("gym_pusht/PushT-v0", render_mode='rgb_array')
        env = env.unwrapped

        # get first observation
        obs, info = env.reset()

        if start_state is not None:
            obs = start_state
            env._set_state(start_state)
            env.agent.position = list(start_state[:2])
            env.block.position = list(start_state[2:4])
            env.block.angle = start_state[4]

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * self.obs_horizon, maxlen=self.obs_horizon)
        # save visualization and rewards
        imgs = [env.render()]
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(obs_seq, stats=self.dataset.stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, self.pred_horizon, self.action_dim), device=self.device)
                    naction = noisy_action

                    # init scheduler
                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    for k in self.noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = self.ema_noise_pred_net(
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=self.dataset.stats['action'])

                # only take action_horizon number of actions
                start = self.obs_horizon - 1
                end = start + self.action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render())

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        print(f'Score: {max(rewards)}; Total steps: {step_idx}')

        return imgs, rewards, step_idx

class GoalConditionedDiffusion():
    def __init__(self, dataset_path,
                 pred_horizon=16, obs_horizon=2,
                 batch_size=256, num_diffusion_iters=100):
        # parameters
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = pred_horizon
        self.batch_size = batch_size
        self.num_diffusion_iters = num_diffusion_iters

        # dataset and data loader
        self.dataset_path = dataset_path
        self.dataset = PushTStateDataset(dataset_path, pred_horizon, obs_horizon, pred_horizon)

        # print data stats
        print("obs.shape:", self.dataset[0]['obs'].shape)
        print("action.shape", self.dataset[0]['action'].shape)
        print("goal.shape", self.dataset[0]['goal'].shape)
        # save data stats
        self.obs_dim = self.dataset[0]['obs'].shape[-1]
        self.action_dim = self.dataset[0]['action'].shape[-1]
        self.goal_dim = self.dataset[0]['goal'].shape[-1]

        # create network
        # conditioned on past obs and an additional goal obs state
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * obs_horizon + self.goal_dim
        )

        # create diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise
            prediction_type='epsilon'
        )

    def to_device(self, device):
        self.device = torch.device(device)
        _ = self.noise_pred_net.to(self.device)

    def train(self, num_epochs=100):
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process after each epoch
            persistent_workers=True)

        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        ema = EMAModel(
            parameters=self.noise_pred_net.parameters(),
            power=0.75)

        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(),
            lr=1e-4, weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * num_epochs
        )

        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = []
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer
                        nobs = nbatch['obs'].to(self.device)
                        naction = nbatch['action'].to(self.device)
                        B = nobs.shape[0]

                        # observation as FiLM conditioning
                        # (B, obs_horizon, obs_dim)
                        obs_cond = nobs[:,:self.obs_horizon,:]
                        # (B, obs_horizon * obs_dim)
                        obs_cond = obs_cond.flatten(start_dim=1)

                        # goal as FiLM conditioning
                        # (B, goal_dim)
                        ngoal = nbatch['goal'].to(self.device)

                        # combine FiLM conditioning
                        # (B, obs_horizon * obs_dim + goal_dim)
                        combined_cond = torch.cat([obs_cond, ngoal], axis=-1)

                        # sample noise to add to actions
                        noise = torch.randn(naction.shape, device=self.device)

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps,
                            (B,), device=self.device
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = self.noise_scheduler.add_noise(
                            naction, noise, timesteps)

                        # predict the noise residual
                        noise_pred = self.noise_pred_net(
                            noisy_actions, timesteps, global_cond=combined_cond)

                        # L2 loss
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        ema.step(self.noise_pred_net.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))

        # Weights of the EMA model
        # is used for inference
        self.ema_noise_pred_net = self.noise_pred_net
        ema.copy_to(self.ema_noise_pred_net.parameters())


    def load_pretrain(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.ema_noise_pred_net = self.noise_pred_net
        self.ema_noise_pred_net.load_state_dict(state_dict)

    def eval(self, start_state=None):
        # load waypoint network
        waypoint_util = WaypointUtil(self.dataset_path, include_agent=True)
        waypoint_util.to_device('cpu')
        waypoint_util.load_pretrain('checkpoint/waypoint_net_with_agent.ckpt')

        # limit enviornment interaction to 200 steps before termination
        max_steps = 200
        env = gym.make("gym_pusht/PushT-v0", render_mode='rgb_array')
        env = env.unwrapped

        # get first observation
        obs, info = env.reset()
        if start_state is not None:
            obs = start_state
            env._set_state(start_state)
            env.agent.position = list(start_state[:2])
            env.block.position = list(start_state[2:4])
            env.block.angle = start_state[4]

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * self.obs_horizon, maxlen=self.obs_horizon)
        # save visualization and rewards
        imgs = [env.render()]
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(obs_seq, stats=self.dataset.stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
                    start_obs = nobs[-1,:]

                    with torch.no_grad():
                        waypoint_util.waypoint_net.eval()
                        # get goal from waypoint network
                        goal = waypoint_util.waypoint_net(start_obs)
                        goal = goal.reshape(1, -1)

                    combined_cond = torch.cat([obs_cond, goal], axis=-1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, self.pred_horizon, self.action_dim), device=self.device)
                    naction = noisy_action

                    # init scheduler
                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    for k in self.noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = self.ema_noise_pred_net(
                            sample=naction,
                            timestep=k,
                            global_cond=combined_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=self.dataset.stats['action'])

                # only take action_horizon number of actions
                start = self.obs_horizon - 1
                end = start + self.action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render())

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        print(f'Score: {max(rewards)}; Total steps: {step_idx}')

        return imgs, rewards, step_idx

if __name__ == '__main__':
    dataset_path = 'data/pusht_cchi_v7_replay.zarr.zip'
    # model = NormalDiffusion(dataset_path)
    # model.to_device('cpu')
    # model.train()
    # torch.save(model.ema_noise_pred_net.state_dict(), './checkpoint/normal_diffusion.ckpt')

    model = GoalConditionedDiffusion(dataset_path)
    model.to_device('cpu')
    model.train()
    torch.save(model.ema_noise_pred_net.state_dict(), './checkpoint/goal_conditioned_diffusion.ckpt')
