import gym
import time
from random import randint

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

# En esta funcion el get state bucketea los estados en funcion de los sizes
# y en funcion de los linspace, por lo tanto si se va a probar es importante conocer los valores usados a la hora de realizar el armado del q values
# el archivo que se levanta hay que cambiarle el nombre

POSITION_SIZE = 2
VELOCITY_SIZE = 2
ANGLE_BINS_SIZE = 12
ANGULAR_VELOCITY_BINS_SIZE = 10
Q = np.zeros((POSITION_SIZE + 1, VELOCITY_SIZE + 1,
             ANGLE_BINS_SIZE + 1, ANGULAR_VELOCITY_BINS_SIZE + 1, 2))

position_bins = np.linspace(-math.inf, math.inf, POSITION_SIZE)
velocity_bins = np.linspace(-math.inf, math.inf, VELOCITY_SIZE)
angle_bins = np.linspace(-0.2095, 0.2095, ANGLE_BINS_SIZE)
angular_velocity_bins = np.linspace(-2.5, 2.5, ANGULAR_VELOCITY_BINS_SIZE)

def get_state(obs):
    # Discretize the state.
    return np.digitize(obs[0], position_bins), np.digitize(obs[1], velocity_bins), np.digitize(obs[2], angle_bins), np.digitize(obs[3], angular_velocity_bins)

def optimal_policy(state, Q):
    action = np.argmax(Q[state])
    return action

def plot(ylabel, data):
    print("Average " + ylabel + ": " + str(np.average(data)))
    print("Min " + ylabel + ": " + str(np.min(data)))
    print("Max " + ylabel + ": " + str(np.max(data)))

def readStoredTrainingResults():
    trainingDataFile = open('training/trainingData', 'rb')
    trainingData = np.load(trainingDataFile)
    if (trainingData.size > 0) :
        Q = trainingData
        print('Q table restored with training saved results.')

start = time.time()

readStoredTrainingResults()

def simulate_qlearning_agent(number_of_samples):
    rewards = [0] * number_of_samples
    for index in range(number_of_samples):
        obs = env.reset()
        done = False
        while not done:
            policy = optimal_policy(get_state(obs), Q)
            obs, reward, done, _ = env.step(policy)
            rewards[index] += reward
            if done:
                env.reset()
    env.close()
    return rewards

number_of_samples = 100000

print("Agent runned with " + str(number_of_samples) +" samples")

rewards = simulate_qlearning_agent(number_of_samples)

plot("Episode", "Rewards", rewards)

end = time.time()

print("------- Timetaken(seconds) ----------------")

print(end-start)