import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym
import gym_pusht

from train_diffusion import NormalDiffusion, GoalConditionedDiffusion

dataset_path = 'data/pusht_cchi_v7_replay.zarr.zip'

# create normal diffusion
normal_diffusion_replan = NormalDiffusion(dataset_path)
normal_diffusion_replan.to_device('cpu')
normal_diffusion_replan.load_pretrain('checkpoint/normal_diffusion.ckpt')

normal_diffusion_no_replan = NormalDiffusion(dataset_path, action_horizon=16)
normal_diffusion_no_replan.to_device('cpu')
normal_diffusion_no_replan.load_pretrain('checkpoint/normal_diffusion.ckpt')

# create goal-conditioned diffusion
waypoint_diffusion = GoalConditionedDiffusion(dataset_path)
waypoint_diffusion.to_device('cpu')
waypoint_diffusion.load_pretrain('checkpoint/goal_conditioned_diffusion.ckpt')

env = gym.make('gym_pusht/PushT-v0', render_mode='rgb_array')
env = env.unwrapped

for test in tqdm(range(3)):
    obs, goal = env.reset()
    # evaluate normal diffusion with replanning
    imgs, rewards_normal_replan, step_idx = normal_diffusion_replan.eval(start_state=obs)
    # save results
    for i, img in enumerate(imgs):
        plt.figure()
        plt.imshow(img)
        plt.savefig(f'img/normal_diffusion_replan/{test}/normal_diffusion_{i}.png')
        plt.close()

    # evaluate normal diffusion without replanning
    imgs, rewards_normal_no_replan, step_idx = normal_diffusion_no_replan.eval(start_state=obs)
    # save results
    for i, img in enumerate(imgs):
        plt.figure()
        plt.imshow(img)
        plt.savefig(f'img/normal_diffusion_no_replan/{test}/normal_diffusion_{i}.png')
        plt.close()

    # evaluate goal conditioned diffusion
    imgs, rewards_waypoint, step_idx = waypoint_diffusion.eval(start_state=obs)
    # save results
    for i, img in enumerate(imgs):
        plt.figure()
        plt.imshow(img)
        plt.savefig(f'img/waypoint_diffusion/{test}/waypoint_diffusion_{i}.png')
        plt.close()

    # save rewards
    rewards_normal_replan = np.array(rewards_normal_replan)
    rewards_normal_no_replan = np.array(rewards_normal_no_replan)
    rewards_waypoint = np.array(rewards_waypoint)
    np.save(f'data/normal_diffusion_replan_rewards_{test}.npy', rewards_normal_replan)
    np.save(f'data/normal_diffusion_no_replan_rewards_{test}.npy', rewards_normal_no_replan)
    np.save(f'data/waypoint_diffusion_rewards_{test}.npy', rewards_waypoint)

    plt.figure()
    plt.plot(rewards_normal_replan, label='Normal Diffusion Replan')
    plt.plot(rewards_normal_no_replan, label='Normal Diffusion No Replan')
    plt.plot(rewards_waypoint, label='Waypoint Diffusion')
    plt.legend()
    plt.title(f'Test {test} Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.savefig(f'img/rewards_{test}.png')
