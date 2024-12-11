import gymnasium as gym
import gym_pusht
import zarr


def replay(datapath):
    env = gym.make("gym_pusht/PushT-v0", render_mode="human")
    env = env.unwrapped
    observation, info = env.reset()

    dataset_root = zarr.open(datapath, mode='r')
    actions = dataset_root['data']['action']
    state = dataset_root['data']['state']
    episode_ends = dataset_root['meta']['episode_ends']


    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        action_sequence = actions[start_idx:episode_ends[i]]
        start_state = state[start_idx]

        env._set_state(start_state)
        env.agent.position = list(start_state[0:2])
        env.block.position = list(start_state[2:4])
        env.block.angle = start_state[4]

        for action in action_sequence:
            observation, reward, terminated, truncated, info = env.step(action)
            image = env.render()

            if terminated or truncated:
                observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    replay('data/pusht_cchi_v7_replay.zarr.zip')
