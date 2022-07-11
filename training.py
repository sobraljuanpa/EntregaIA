#pip install gym pygame
import gym
import time
from random import randint

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

# Declare constants for buckets.
POSITION_SIZE = 2
VELOCITY_SIZE = 2
ANGLE_BINS_SIZE = 6
ANGULAR_VELOCITY_BINS_SIZE = 6

position_bins = np.linspace(-0.2, 0.2, POSITION_SIZE)
velocity_bins = np.linspace(-1,1, VELOCITY_SIZE)
angle_bins = np.linspace(-0.25, 0.25, ANGLE_BINS_SIZE)
angular_velocity_bins = np.linspace(-1.5, 1.5, ANGULAR_VELOCITY_BINS_SIZE)

discount_factor = 0.9

env = gym.make('CartPole-v1')


def epsilon_greedy_policy(state, Q, epsilon):
    if np.random.random() < epsilon:
        action = randint(0, 1)
    else:
        action = np.argmax(Q[state])
    return action


def optimal_policy(state, Q):
    action = np.argmax(Q[state])
    return action


def get_state(obs):
    # Discretize the state.
    return np.digitize(obs[0], position_bins), np.digitize(obs[1], velocity_bins), np.digitize(obs[2], angle_bins), np.digitize(obs[3], angular_velocity_bins)


Q = np.zeros((POSITION_SIZE + 1, VELOCITY_SIZE + 1,
             ANGLE_BINS_SIZE + 1, ANGULAR_VELOCITY_BINS_SIZE + 1, 2))


def update_q(state, action, reward, new_state):
    Q[state][action] += (learning_rate * (reward +
                         discount_factor * np.max(Q[new_state]) - Q[state][action]))

def saveTrainingResult():
    trainingDataFile = open('training/trainingDataJP', 'wb')
    np.save(trainingDataFile, Q)
    trainingDataFile.close

def readStoredTrainingResults():
    trainingDataFile = open('training/trainingDataJP', 'rb')
    trainingData = np.load(trainingDataFile)
    if (trainingData.size > 0) :
        Q = trainingData
        print('Q table restored with training saved results.')

start = time.time()

# Params for training.
epsilon = 0.1
learning_rate = 0.9

episodes_of_training = 1000000
# Array to analyze the training.
steps = [0] * episodes_of_training

for episode in range(episodes_of_training):
    obs = env.reset()
    current_state = get_state(obs)
    done = False
    while not done:
        steps[episode] += 1
        # Choose action
        action = epsilon_greedy_policy(current_state, Q, epsilon)
        # Update action
        obs, reward, done, info = env.step(action)
        new_state = get_state(obs)
        # Update Q.
        update_q(current_state, action, reward, new_state)
        current_state = new_state
print("I have trained with " + str(episodes_of_training) + " iterations")
end = time.time()
print("------- Training Timetaken(seconds) ----------------")
print(end-start)

print("Average stepe: " + str(np.average(steps)))
print("Min steps " + str(np.min(steps)))
print("Max steps " + str(np.max(steps)))

saveTrainingResult()

readStoredTrainingResults()
def simulate_random_agent(number_of_samples):
    print(Q)
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

start = time.time()

number_of_samples = 10000

print("Agent runned with " + str(number_of_samples) +" samples")

rewards = simulate_random_agent(number_of_samples)

end = time.time()
print("------- Timetaken(seconds) ----------------")
print(end-start)


print("Average rewards: " + str(np.average(rewards)))
print("Min rewards" + str(np.min(rewards)))
print("Max rewards" + str(np.max(rewards)))