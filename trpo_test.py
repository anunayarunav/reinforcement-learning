import gym
from gym import wrappers
import matplotlib.pyplot as plt

def play(num_episodes, num_steps, policy, update=None, preprocess = None, num_obs=1):
    time_steps = []
    for i_episode in range(num_episodes):
        observation = env.reset()
        states, actions, rewards = [], [], []
        observations = []
        r = 0
        for t in range(num_steps):
            #env.render()
            #observation = np.concatenate((np.array([observation[0]]),observation[1]))
            
            states.append(observation)
            
            if len(states) < num_obs:
                action = env.action_space.sample()
            else:
                obs = states[-num_obs:]
                
                if preprocess is not None:
                    for i in range(num_obs):
                        obs[i] = preprocess(obs[i])
                
                observations.append(np.array(obs))
                action = policy(np.array(obs))

            observation, reward, done, info = env.step(action)
            r += reward
            
            actions.append(action)
            rewards.append(reward)
                
            if done:
                break
        
        if update:
            update(actions[num_obs-1:], observations, rewards[num_obs-1:])
        
        time_steps.append(t)

    #env.close()
    #w, b = best_params
    return time_steps, r

#%load_ext autoreload 
#%autoreload 2
#%matplotlib inline


def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    
    return processed_observation.reshape(80, 80, 1)

def generate_observation(observation1, observation2, observation3, observation4):
    return np.dstack((observation1, observation2, observation3, observation4))

from lib.TRPO import *
import matplotlib.pyplot as plt

env = gym.make("Pong-v0")
#env = wrappers.Monitor(env, '/tmp/acrobot-v1',force=True)
#bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k //tensorflow/tools/pip_package:build_pip_package

d = 9*9*3
bins = 2
num_obs = 2
input_shape = (None, num_obs, 80, 80, 1)
layers = ['flatten', 'fc', (None, 256)]
discount = 0.99
T = 5000

trpo = TRPO(input_shape, layers, num_actions=bins)
plot_rews = []

avg_rew = 0
alpha = 0.99
batch_size = 32

def update(actions, states, rewards):
    
    acc_rew = []
    n = len(rewards)
    if n == 0:
        return
    
    acc_rew.append(rewards[-1])
    for i in range(n-1):
        acc_rew = [discount*acc_rew[0] + rewards[n-i-1]] + acc_rew
    
    #for i in range(n):
    #    acc_rew[i] -= acc_rew[i]/(n-i)
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions)
    advantages = np.array(acc_rew)
    
    #split_states = np.array_split(states, batch_size)
    #split_actions = np.array_split(actions, batch_size)
    #split_advantages = np.array_split(advantages, batch_size)
    #for st, ac, ad in zip(split_states, split_actions, split_advantages):
    #    trpo.add_trajectory(st, ac, ad)
    
    trpo.add_trajectory(states, actions, advantages)
    
    global avg_rew
    avg_rew = alpha*avg_rew + (1-alpha)*np.sum(rewards)
    
    print(np.sum(rewards), avg_rew)
    
    plot_rews.append(avg_rew)
    
    if len(plot_rews) % 20 == 0:
        print(avg_rew, np.sum(rewards))
    
    if len(plot_rews)%100 == 0:
        plt.clf()
        plt.plot(plot_rews[1500:])
        plt.show()
    
def policy(obs):
    return trpo.choose_action(obs)[0] + 2

play(5000, T, policy, update, preprocess=preprocess_observations, num_obs=num_obs)
