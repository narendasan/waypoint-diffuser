import numpy as np
import matplotlib.pyplot as plt

for test in range(3):
    waypoint_rewards = np.load(f'data/waypoint_diffusion_rewards_{test}.npy')
    normal_rewards = np.load(f'data/normal_diffusion_replan_rewards_{test}.npy')

    plt.figure()
    plt.plot(waypoint_rewards, label='Waypoint Diffusion')
    plt.plot(normal_rewards, label='Normal Diffusion')
    plt.legend()
    plt.title(f'Test {test} Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.savefig(f'img/rewards_partial_{test}.png')
