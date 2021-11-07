# Group Members
# Jenalea Rajab: 562262
# Amy Pegram: 1825142
# Claudio Surmon: 1830290
# Rushil Daya: 1830490

# The following tutorials/public gits were used for the implementation of this Assignment:
# References
# [1] A. Paszke, “Reinforcement learning (dqn) tutorial.” [Online]. Available: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# [2] J. TORRES.AI. Deep q-network (dqn)-i. [Online]. Available: https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af
# [3] K. Tessera. Dqn atari. [Online]. Available: https://github.com/KaleabTessera/DQN-Atari
# [4] https://github.com/Pieter-Cawood/Reinforcement-Learning/blob/master/NLE_DQN/Agent.py
# [5] The skeleton code for the COMS4047A/COMS7053A Lab 3 - Deep Q-Network

import random
from minihack import reward_manager
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
import skimage.io as io
from minihack import RewardManager
import random
import os

if __name__ == "__main__":
    frame = 0
    hyper_params = {
        "seed": random.randint(0,1000),  # which seed to use
        "env": "MiniHack-Quest-Hard-v0",  # name of the game
        "replay-buffer-size": int(2e3),  # replay buffer size
        "learning-rate": 0.99,  # learning rate for RMSprop optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.1,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
    }
    #Actions used by the agent to navigate the world
    MOVE_ACTIONS = tuple(nethack.CompassDirection)

    #More actions the action can do
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.PICKUP, #the agent can pickup items that are then stored in the inventory
        nethack.Command.INVENTORY, #an agent can use items in its inventory, such as leviation boots to cross the lava pool
        nethack.Command.LOOK, #an agent can look what is here
        nethack.Command.OPEN, # an agent can open a door
    )   

    #Function used to simplify and normalize the observation space (based on code from [4])
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

        #translate all the text characters to ASCII
        walls = ord('-')
        doors = ord('|')
        closed_door = ord('+')
        corridor = ord('#')
        lava = ord('}')
        monster = ord('I')
        demon = ord('&')

        #create a copy of the observation space
        copy_obs = observation['chars_crop']

        #create numpy array to represent our observation space
        obs_chars = np.zeros(copy_obs.shape) #set everything to 0
        obs_chars[np.where((copy_obs == lava) & (copy_obs == monster) & (copy_obs == demon))] = 0.2 #set hostile objects to 0.2
        obs_chars[np.where((copy_obs == walls) | (copy_obs == doors) | copy_obs == closed_door | (copy_obs == corridor))] = 0.5 #set environment to 0.5

        return obs_chars

    #get the seed from the given hyperparameters
    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    #Change the built in reward manager to include a reward for opening a door and killing a demon
    reward_gen = RewardManager()
    reward_gen.add_location_event("door", reward=0.2)
    reward_gen.add_kill_event("demon", reward=0.2)

    #create the env, with glyphs_crop and not glyphs so that observation space will be 9x9
    env = gym.make(hyper_params["env"],  observation_keys=("glyphs_crop", "chars_crop", "colors", "pixel", "blstats"), actions = NAVIGATE_ACTIONS, reward_manager = reward_gen)
    env.seed(hyper_params["seed"]) #environment is created with the random seed
    action_space = env.action_space

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(env.observation_space["glyphs_crop"], env.action_space, replay_buffer,
                    hyper_params["learning-rate"],
                    hyper_params["batch-size"],
                    hyper_params["discount-factor"])

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    episode_loss = []

    state = env.reset() #reset the state space
    glyphs = format_observations(state) #format the observation space for normalization

    # Episode loop taken from the Lab [5]
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps) 
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()

        if sample <= eps_threshold:
            action = env.action_space.sample()
        else:
            action = agent.act(glyphs)

        # Take step in env
        next_state, reward, done, _ = env.step(action)
        
        # Add state, action, reward, next_state, float(done) to reply memory - cast done to float
        done = float(done)

        glyph_next_state = format_observations(next_state) # format observation before adding to the replay buffer
        agent.replay_buffer.add(glyphs,action,reward,glyph_next_state,done)
        
        # Update the state
        state = next_state

        # Add reward to episode_reward
        episode_rewards[-1] += reward

        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        #if the program has run for a certain amount of steps, the neural will start learning using the replay buffer
        if (t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0): 
            episode_loss.append(agent.optimise_td_loss()) #parameters are updated every 5 steps

        if (t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0):
            agent.update_target_network() #update the target network every 1000 steps

        num_episodes = len(episode_rewards)
        #Use the pixel parameter to create a video for our agent at the last step of the epsiode
        if num_episodes > 900 and num_episodes <= 901:
            if not os.path.isdir(f"video_" + str(hyper_params["seed"])):
                os.mkdir(f"video_" + str(hyper_params["seed"]))

            io.imsave(f"video_"+ str(hyper_params["seed"]) +f"/frame_{frame}.png",next_state["pixel"])
            frame += 1
        
        if (done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params["print-freq"] == 0):
            #Save the reward and the loss
            np.savetxt('rewards_'+ str(hyper_params["seed"]) +'.csv', episode_rewards, delimiter=',', fmt='%1.5f')
            np.savetxt('loss_'+ str(hyper_params["seed"]) +'.csv', episode_loss,delimiter=',', fmt='%1.5f')

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
    torch.save(agent.policy_network, 'model'+ str(hyper_params["seed"]) +'.pt')
