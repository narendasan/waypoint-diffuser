import torch
import numpy as np
import gymnasium as gym
import gym_pusht

import matplotlib.pyplot as plt
from train_waypoint import WaypointUtil
from push_t_dataset import normalize_data, unnormalize_data

# load pre-trained model
waypoint_util = WaypointUtil('data/pusht_cchi_v7_replay.zarr.zip')
waypoint_util.to_device('cpu')
waypoint_util.load_pretrain('checkpoint/waypoint_net.ckpt')

env = gym.make('gym_pusht/PushT-v0', render_mode='rgb_array')
env = env.unwrapped

for test in range(10):
    # get first observation
    obs, info = env.reset()
    # get rid of the agent
    env.agent.position = [0, 0]
    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    start = env.render()

    plt.figure()
    plt.imshow(start)
    plt.savefig(f'./img/waypoint/waypoint_{test}_0')

    for step in range(10):
        # get goal from waypoint network
        obs_input = normalize_data(torch.tensor(obs).float(),
                                   waypoint_util.dataset.stats['obs'])
        waypoint = waypoint_util.waypoint_net(obs_input[2:]).detach().numpy()
        waypoint_full = np.zeros((5))
        waypoint_full[2:] = waypoint
        waypoint_full = unnormalize_data(waypoint_full, waypoint_util.dataset.stats['obs'])

        # override T position
        env.block.position = list(waypoint_full[2:4])
        env.block.angle = waypoint_full[4]
        # step the environment without moving the agent
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

        state = env.render()
        plt.figure()
        plt.imshow(state)
        plt.savefig(f'./img/waypoint/waypoint_{test}_{step+1}')
