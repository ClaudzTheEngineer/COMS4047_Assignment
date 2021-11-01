import random
import numpy as np
import gym
import torch

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import time
import minihack 
from nle import nethack
import copy

if __name__ == "__main__":

    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "MiniHack-Quest-Hard-v0",  # name of the game
        "replay-buffer-size": int(2e3),  # replay buffer size
        "learning-rate": 0.99,  # learning rate for RMSprop optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(3e6),  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.1,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
        "mode": "train"

    }

    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.PICKUP,
        nethack.Command.INVENTORY,
        nethack.Command.SEARCH,
    )   

    STATS_IDX = {
        'x_coordinate': 0,
        'y_coordinate': 1,
        'score': 9,
    }

    def format_observations(observation):
        # - and | The walls of a room, or an open door. Or a grave (|).
        # . The floor of a room, ice, or a doorless doorway.
        # # A corridor, or iron bars, or a tree, or possibly a kitchen sink (if your dungeon has
        # sinks), or a drawbridge.
        # > Stairs down: a way to the next level.
        # < Stairs up: a way to the previous level.
        # + A closed door, or a spellbook containing a spell you may be able to learn.
        # @ Your character or a human.
        # $ A pile of gold.
        # ^ A trap (once you have detected it).
        # ) A weapon.
        # [ A suit or piece of armor.
        # % Something edible (not necessarily healthy).
        # ? A scroll.
        # / A wand.
        # = A ring.
        # ! A potion.
        # ( A useful item (pick-axe, key, lamp . . . ).
        # " An amulet or a spider web.
        # * A gem or rock (possibly valuable, possibly worthless).
        # ` A boulder or statue.
        # 0 An iron ball.
        # _ An altar, or an iron chain.
        # { A fountain.
        # } A pool of water or moat or a pool of lava.
        # \ An opulent throne
        # I This marks the last known location of an invisible or otherwise unseen monster. Note that the monster could have moved

        walls = ord('-')
        doors = ord('|')
        closed_door = ord('+')
        #corridor = ord('#')
        lava = ord('}')
        monster = ord('I')

        copy_obs = observation['chars']
        obs_chars = np.zeros(copy_obs.shape)
        obs_chars[np.where((copy_obs == lava) & (copy_obs == monster))] = 0.2
        obs_chars[np.where((copy_obs == walls) | (copy_obs == doors) | copy_obs == closed_door)] = 0.5

        x_loc = observation['blstats'][STATS_IDX['x_coordinate']]
        y_loc = observation['blstats'][STATS_IDX['y_coordinate']]
        score = observation['blstats'][STATS_IDX['score']]

        obs_stats = np.array([x_loc,y_loc,score])
        return obs_chars, obs_stats.astype(np.float32)

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    env = gym.make(hyper_params["env"],  observation_keys=("glyphs_crop", "chars", "colors", "pixel"), actions = NAVIGATE_ACTIONS)
    env.seed(hyper_params["seed"])
    action_space = env.action_space

    # Call the gym wrapper to create the video
    #env = gym.wrappers.Monitor(env, './minihack_video/', video_callable=lambda episode_id: episode_id % 100 == 0,force=True)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(env.observation_space["glyphs_crop"], env.action_space, replay_buffer,
                     hyper_params["learning-rate"],
                     hyper_params["batch-size"],
                     hyper_params["discount-factor"])

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    episode_loss = []

    state = env.reset()
    print(state)
    glyphs,stats = format_observations(state)
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()

        if sample <= eps_threshold:
            action = env.action_space.sample()
        else:
            action = agent.act(glyphs,stats)

        # Take step in env
        next_state, reward, done, _ = env.step(glyphs,stats)
        # Add state, action, reward, next_state, float(done) to reply memory - cast done to float
        done = float(done)

        agent.replay_buffer.add(state, action, reward, next_state, done)
        # Update the state
        state = next_state

        # Add reward to episode_reward
        episode_rewards[-1] += reward


        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            episode_loss.append(agent.optimise_td_loss())

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            #Save the reward and the loss
            np.savetxt('rewards.csv', episode_rewards, delimiter=',', fmt='%1.5f')
            np.savetxt('loss.csv', episode_loss,delimiter=',', fmt='%1.5f')

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
    torch.save(agent.policy_network, 'model.pt')
