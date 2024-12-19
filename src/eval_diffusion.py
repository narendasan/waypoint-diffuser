import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym
import gym_pusht

from train_diffusion import NormalDiffusion, GoalConditionedDiffusion

dataset_path = 'data/pusht_cchi_v7_replay.zarr.zip'

# create normal diffusion
normal_diffusion = NormalDiffusion(dataset_path)
normal_diffusion.to_device('cpu')
normal_diffusion.load_pretrain('checkpoint/normal_diffusion.ckpt')

# create goal-conditioned diffusion
goal_diffusion = GoalConditionedDiffusion(dataset_path)
goal_diffusion.to_device('cpu')
goal_diffusion.load_pretrain('checkpoint/goal_conditioned_diffusion.ckpt')

env = gym.make('gym_pusht/PushT-v0', render_mode='rgb_array')
env = env.unwrapped

for test in tqdm(range(3)):
    obs, goal = env.reset()
    # evaluate normal diffusion
    imgs, rewards, step_idx = normal_diffusion.eval(start_state=obs)
    # save results
    plt.figure()
    plt.plot(rewards)
    plt.savefig(f'img/normal_diffusion/{test}/normal_diffusion_rewards.png')
    plt.close()

    for i, img in enumerate(imgs):
        plt.figure()
        plt.imshow(img)
        plt.savefig(f'img/normal_diffusion/{test}/normal_diffusion_{i}.png')
        plt.close()

    # evaluate goal conditioned diffusion
    imgs, rewards, step_idx = goal_diffusion.eval(start_state=obs)
    # save results
    plt.figure()
    plt.plot(rewards)
    plt.savefig(f'img/goal_diffusion/{test}/goal_diffusion_rewards.png')
    plt.close()

    for i, img in enumerate(imgs):
        plt.figure()
        plt.imshow(img)
        plt.savefig(f'img/goal_diffusion/{test}/goal_diffusion_{i}.png')
        plt.close()
