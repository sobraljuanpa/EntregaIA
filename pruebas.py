#pip install gym pygame
import gym
import time
from random import randint
# imports personales
import numpy as np

env = gym.make('CartPole-v1')

print("Environment observation space")
print(env.observation_space)


print("Environment action space")
print(env.action_space)
# 0 <- izquierda, 1 -> derecha

rewards = [0] * 10#inicializo lista en 0 por 10 lugares

for i in range(10):
    obs = env.reset()
    done = False
    while not done:
        policy = randint(0,1)
        obs, reward, done, info = env.step(policy)
        rewards[i] += reward
        print(obs)
        env.render()
        time.sleep(0.05)
        if done:
            env.reset()
env.close()

print("Rewards")
for x in range(10):
    print(rewards[x])
