# Group Members
# Jenalea Rajab: 562262
# Amy Pegram: 1825142
# Claudio Surmon: 1830290
# Rushil Daya: 1830490

# The following tutorials/public gits were used for the implementation of this Lab

#  [1]https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/12/REINFORCE-CartPole.html#REINFORCE
#  [2]https://github.com/KaleabTessera/Policy-Gradient/blob/master/reinforce/reinforce.py
#  [3]https://github.com/andrecianflone/rl_at_ammi/blob/master/REINFORCE_solution.ipynb


import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
from torch.distributions import Categorical

import random
from collections import deque
import minihack 
from nle import nethack

#Using a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Policy 
# The output of the NN is the action and the log probability of the action 
class SimplePolicy(nn.Module):
    def __init__(self, s_size,  a_size, h_size=16):
        #print(f"size in action: {a_size}")
        super(SimplePolicy,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        #For the 84x84 input, the output from the convolution layer will have 3136
        #9x9 -> 7x7 -> 5x5
        #5x5x32
        self.fc1 = nn.Linear(576, a_size)
        #self.fc2 = nn.Linear(512, a_size)

    def forward(self, x):
        #Where x is the state
        #print(x.shape)
        y = np.array(x).astype('float')
        x_glyphs = torch.from_numpy(y).unsqueeze(0).float().to(device)
        x_glyphs = x_glyphs.unsqueeze(0).float().to(device)
        #print(x_glyphs.shape)
        #x_glyphs = torch.transpose(x_glyphs,1,3)
        #x_glyphs = torch.transpose(x_glyphs,0,2)
        # Implement the Deep Q-Network
        x = nn.functional.relu(self.conv1(x_glyphs))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        #Flatten the 4D tensor (bastch_size x color_channel x stack x dimensions) to 2D tensor
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        
        #We want the 1D probability of the action
        action_probs = nn.functional.softmax(x, dim = 1)
        #print(f"size in action after probs: {action_probs.shape}")
        
        #Categorical distribution over all possible actions 
        categorical_dist = Categorical(action_probs)
        
        #Sample the action from the distribution 
        action = categorical_dist.sample()

        #Return the action and the log_probability of the action 
        return action.item(), categorical_dist.log_prob(action)
        
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

        copy_obs = observation['chars_crop']

        obs_chars = np.zeros(copy_obs.shape)
        obs_chars[np.where((copy_obs == lava) & (copy_obs == monster))] = 0.2
        obs_chars[np.where((copy_obs == walls) | (copy_obs == doors) | copy_obs == closed_door)] = 0.5

        x_loc = observation['blstats'][STATS_IDX['x_coordinate']]
        y_loc = observation['blstats'][STATS_IDX['y_coordinate']]
        score = observation['blstats'][STATS_IDX['score']]

        obs_stats = np.array([x_loc,y_loc,score])

        return obs_chars, obs_stats

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    #From the sequence of rewards we can calculate the return
    G = 0
    
    #for each step in the trajectory get the reward and calculate the return
    for t in range(len(rewards)):
        G += gamma**t * rewards[t]
    
    return G


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    
    #Set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)  
    
    #Initalize the policy and Adam optimizer
    policy = policy_model.to(device)
    Adam_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    #All Rewards
    scores = []
    #scores_deque = deque(maxlen=100)
    
    #For each episode
    for episode in range(1, number_episodes+1):
        #Initalize rewards and log prob arrays
        log_probabilities = []
        rewards_all = []
        
        #Reset the Environment
        state = env.reset()
        glyphs,_ = format_observations(state)
        #Generate the episode and save the reward and log probability
        for step in range(max_episode_length):
            
            #Get the action and log prob from the policy
            action, action_prob = policy.forward(glyphs)
            
            #print(f"action: {action}")
            #Take the action and get the next_state, reward and done
            next_state, reward, done, _ = env.step(action)
            
            #Store the reward and log probability of the action
            rewards_all.append(reward)
            log_probabilities.append(action_prob)
            
            #If the episode is done break
            if done == True:
                break
            
            state = next_state
        
        #Save the total reward for the episode
        scores.append(sum(rewards_all))
        #scores_deque.append(sum(rewards_all))
        
        #Compute the Return
        returns_G = compute_returns(rewards_all, gamma)
        
        
        #Calculate the loss-----------------------------
        loss = []
        
        #Multiply the log probabilities by the returns and sum
        #We are using gradient ascent and we want to maximise the returns so we multiply by -1
        for prob in log_probabilities:
            loss.append(-prob * returns_G)
       
        loss = torch.cat(loss).sum()
        
        #Update the parameters----------------------------

        Adam_optimizer.zero_grad()
        loss.backward()
        Adam_optimizer.step()

        # report the score to check that we're making progress [2]
        #if episode % 50 == 0 and verbose:
        print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores)))

    return policy_model, scores


def compute_returns_naive_baseline(rewards, gamma):
    #Naive approach using the average of rewards over a single trajectory
    # and normalise the rewards by dividing by the std deviation
    cumulative_rewards = 0
    G = []
    
    #Present actions should only impact the future
    for t in reversed(range(len(rewards))):
        cumulative_rewards =  gamma * cumulative_rewards + rewards[t]
        G.insert(0, cumulative_rewards)
        
    G = np.array(G)
    mean = G.mean(axis=0)
    std = G.std(axis=0)
    G = (G - mean)/std
    return G


def reinforce_naive_baseline(env, policy_model, seed, learning_rate,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    
    #Initalize the policy and Adam optimizer
    policy = policy_model.to(device)
    Adam_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    #All Rewards
    scores = []
    #scores_deque = deque(maxlen=100)
    
    #For each episode
    for episode in range(1, number_episodes+1):
        
        #Initalize rewards and log prob arrays
        log_probabilities = []
        rewards_all = []
        
        #Reset the Environment
        state = env.reset()
        glyphs,_ = format_observations(state)
        #Generate the episode and save the reward and log probability
        for step in range(max_episode_length):
            
            #Get the action and log prob from the policy
            action, action_prob = policy.forward(glyphs)
            
            #Take the action and get the next_state, reward and done
            next_state, reward, done, _ = env.step(action)
            
            #Store the reward and log probability of the action
            rewards_all.append(reward)
            log_probabilities.append(action_prob)
            
            #If the episode is done break
            if done == True:
                break
            
            state = next_state
        
        #Save the total reward for the episode
        scores.append(sum(rewards_all))
        #scores_deque.append(sum(rewards_all))
        
        #Compute the Return [2]
        returns_G = compute_returns_naive_baseline(rewards_all, gamma)
        returns_G = torch.from_numpy(returns_G).float().to(device)
        
        #Calculate the loss-----------------------------
        loss = []

        log_probabilities = torch.cat(log_probabilities)
        #Multiply the log probabilities by the returns and sum
        #We are using gradient ascent and we want to maximise the returns so we multiply by -1
        loss = -torch.sum(log_probabilities*returns_G)
        
        #Update the parameters----------------------------

        Adam_optimizer.zero_grad()
        loss.backward()
        Adam_optimizer.step()

        # report the score to check that we're making progress [2]
        if episode % 50 == 0 and verbose:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores)))

    return policy, scores

def run_reinforce(env):
    # Instantiate the Environment
    # The env has a discrete action space (2) ( left or right ), 
    # and a 4 dimensional state space (position, velocity, pendulum angle and its angular velocity)
    
    policy_model = SimplePolicy(s_size=env.observation_space["glyphs_crop"].shape[0], a_size=env.action_space.n, h_size=50)
    policy, scores = reinforce(env=env, policy_model=policy_model, seed=42, learning_rate=1e-2,
                               number_episodes=50,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
    
    # Plot learning curve [1]
    
    moving_av = moving_average(scores, 50)
    plt.plot(scores, label='Score')
    plt.plot(moving_av, label='Moving Average (w=50)', linestyle='--')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('REINFORCE learning curve for MiniHack')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.show()
    


def investigate_variance_in_reinforce(env):
    #The variance can be seen by running multiple trials with different seeds and averaging the results
    
    #Create the environment and set 5 random seeds
    seeds = np.random.randint(1000, size=5)
    
    run_scores = []
    run = 0 
    
    #Initialise the policy and run the reinforce algorithm
    for s in seeds:
        policy_model = SimplePolicy(s_size=env.observation_space.shape[0], a_size=env.action_space.n,h_size=50)
        policy, scores = reinforce(env=env, policy_model=policy_model, seed=int(s), learning_rate=1e-2,
                               number_episodes=50,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=False)
        run_scores.append(scores)
        
        run+=1
        
        print(f"Run:  {run}  complete.")
    
    #Computing the mean and std deviation of returns for all runs [2]
    moving_avg_scores = np.array([moving_average(score, 1) for score in run_scores])
    mean = moving_avg_scores.mean(axis=0)
    std = moving_avg_scores.std(axis=0)

    plt.plot(mean, '-', color='blue')
    x_axis = np.arange(1, len(mean)+1)
    plt.fill_between(x_axis, mean-std, mean+std, color='blue', alpha=0.2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Q1 REINFORCE averaged over 5 seeds')
    plt.savefig('Q1 reinforce_average.png')
    plt.show()
    

    return mean, std


def run_reinforce_with_naive_baseline(env, mean, std):
    
    #Create the environment and set 5 random seeds

    np.random.seed(53)
    seeds = np.random.randint(1000, size=5)
    
    run_scores = []
    run = 0 
    
    #Initialise the policy and run the reinforce algorithm
    for s in seeds:
        policy_model = SimplePolicy(s_size=env.observation_space.shape[0], a_size=env.action_space.n,h_size=50)
        policy, scores = reinforce_naive_baseline(env=env, policy_model=policy_model, seed=int(s), learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=False)
        run_scores.append(scores)
        
        run+=1
        
        print(f"Reinforce with Naive Baseline Run:  {run}  complete.")
    
    #Computing the mean and std deviation of returns for all runs [2]
    moving_avg_scores_naive = np.array([moving_average(score, 50) for score in run_scores])
    mean_naive = moving_avg_scores_naive.mean(axis=0)
    std_naive = moving_avg_scores_naive.std(axis=0)
    
    
    #Reinforce (from mean and std)
    plt.plot(mean, '-', color='blue', label = 'Reinforce')
    x_axis = np.arange(1, len(mean)+1)
    plt.fill_between(x_axis, mean-std, mean+std, color='blue', alpha=0.2)
    
    #Reinforce with naive baseline
    plt.plot(mean_naive, '-', color='green', label = 'Reinforce with naive baseline')
    x_axis = np.arange(1, len(mean_naive)+1)
    plt.fill_between(x_axis, mean_naive-std_naive, mean_naive+std_naive, color='green', alpha=0.2)
    
    
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Q2 REINFORCE and REINFOCE with Naive Baseline averaged over 5 seeds')
    plt.savefig('Q2 reinforce_average.png')
    plt.show()
    
    return

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

if __name__ == '__main__':
    env = gym.make('MiniHack-Quest-Hard-v0',observation_keys=("glyphs_crop", "chars_crop", "colors", "pixel", "blstats"), actions = NAVIGATE_ACTIONS)
    run_reinforce(env)
    mean, std = investigate_variance_in_reinforce(env)
    #run_reinforce_with_naive_baseline(env, mean, std)
