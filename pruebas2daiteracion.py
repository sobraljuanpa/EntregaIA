#pip install gym pygame
import gym
import time
from random import randint
# imports personales
import numpy as np

env = gym.make('CartPole-v1')

def epsilon_greedy_policy(state, Q, epsilon):
    if np.random.random() < epsilon:
        action = randint(0,1)
    else:
        action = np.argmax(Q[state])
    return action

def optimal_policy(state, Q):
    action = np.argmax(Q[state])
    return action

angular_velocity_bins = np.linspace(-2.5, 2.5, 10)
angle_bins = np.linspace(-0.2095, 0.2095, 12) #armamos los bins en los q clasificar observaciones

def get_state(obs):
    return np.digitize(obs[2], angle_bins), np.digitize(obs[3], angular_velocity_bins) #nos dice en que bin cae cada observacion

Q = np.zeros((13, 11, 2))

# #defino params de entrenamiento
epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9

def update_q(state, action, reward, new_state):
        """
        Updates Q-table using the rule as described by Sutton and Barto in
        Reinforcement Learning.
        """
        Q[state][action] += (learning_rate * (reward + discount_factor * np.max(Q[new_state]) - Q[state][action]))

start = time.time()

for episode in range(50000):
    obs = env.reset()
    current_state = get_state(obs)
    done = False
    while not done:
        #elijo acccion
        action = epsilon_greedy_policy(current_state, Q, epsilon)
        #actualizo accion
        obs, reward, done, info = env.step(action)
        new_state = get_state(obs)
        #actualizo q
        update_q(current_state, action, reward, new_state)
        current_state = new_state
print("Me entrene con 100000 iteraciones")
print(Q)



def simulate_random_agent(number_of_samples):
    rewards = [0] * number_of_samples #inicializo lista para almacenar recompensas
    for index in range(number_of_samples):
        obs = env.reset()
        done = False
        while not done:
            policy = optimal_policy(get_state(obs), Q)
            obs, reward, done, info = env.step(policy)
            rewards[index] += reward
            if done:
                env.reset()
    env.close()
    average = np.average(rewards)

    return average, rewards

print("Simulacion entrenada 10000 muestreos")

print("Valor esperado :")
print(simulate_random_agent(10000)[0])
end = time.time()
print("------- Timetaken(seconds) ----------------")
print(end-start)



# Se agrega una segunda dimension, si se entrena con 1000 episodios son pocos para la cantidad de posibles estados (pasa de 12 a 120)
# y muestra performance promedio peor que la iteracion previa solamente con 12 estados.
# Se arranca a entrenar con 10000 episodios y tomar muestreos promedio del valor de la policy con 10000 episodios tambien.
# Se arrancan a ver valores arriba de 22 con constancia (con 10 bins, rango de -5 a 5), pero se ven muchos bins vacios o semi vacios

# Se sugiere probar cambiar el rango a valores mas cercanos a los observados (-2.5 a 2.5) y aumentra la cant de episodios.
# Con un espacio de 10 para la posicion, 5 para las velocidades como descrito rpeviamente, con 10k episodios de entrenamiento
# el valor esperado es de aprox 150

