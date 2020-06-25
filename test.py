import numpy as np
import torch
import gym
import os
import time

from agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Wrapper():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, env):
        self.env = env  

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = env.reset()
        img_gray = rgb2gray(img_rgb)
        self.stack = [img_gray] * img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(action_repeat):
            img_rgb, reward, die, _ = env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == img_stack
        return np.array(self.stack), total_reward, done, die


    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
    
agent = Agent(device)

env_wrap = Wrapper(env)


# 1.Prepare load
def load(agent, directory, filename):
    agent.net.load_state_dict(torch.load(os.path.join(directory,filename)))


from collections import deque
import os


# 2.Prepare player 
def play(env, agent, n_episodes):
    state = env_wrap.reset()
    
    scores_deque = deque(maxlen=100)
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        state = env_wrap.reset()        
        score = 0
        
        time_start = time.time()
        
        while True:
            action, a_logp = agent.select_action(state)
            env.render()
            next_state, reward, done, die = env_wrap.step( \
                action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))

            state = next_state
            score += reward
            
            if done or die:
                break 

        s = (int)(time.time() - time_start)
        
        scores_deque.append(score)
        scores.append(score)

        print('Episode {}\tAverage Score: {:.2f},\tScore: {:.2f} \tTime: {:02}:{:02}:{:02}'\
                  .format(i_episode, np.mean(scores_deque), score, s//3600, s%3600//60, s%60))


# 3.Load and play 

load(agent, 'dir_chk', 'model_weights_0.pth')
play(env, agent, n_episodes=5)



load(agent, 'dir_chk', 'model_weights_400.pth')
play(env, agent, n_episodes=5)



load(agent, 'dir_chk', 'model_weights_600.pth')
play(env, agent, n_episodes=5)


load(agent, 'dir_chk', 'model_weights_800.pth')
play(env, agent, n_episodes=5)

load(agent, 'dir_chk', 'model_weights_830.pth')
play(env, agent, n_episodes=5)


load(agent, 'dir_chk', 'model_weights_870.pth')
play(env, agent, n_episodes=5)


load(agent, 'dir_chk', 'model_weights_900.pth')
play(env, agent, n_episodes=5)


env.close()



