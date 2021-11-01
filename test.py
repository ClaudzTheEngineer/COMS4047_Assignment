import gym
import minihack
import numpy as np
import pdb

env = gym.make("MiniHack-Quest-Hard-v0", observation_keys=("glyphs", "chars", "colors", "pixel"))
env.reset() # each reset generates a new environment instance
env.render()
next_state, reward, done, obs = env.step(1) 
env.render()
next_state, reward, done, obs = env.step(2) 
env.render()
a = np.unique(next_state["chars"])
pdb.set_trace()