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

bins = np.linspace(-0.2095, 0.2095, 12) #armamos los bins en los q clasificar observaciones

def get_state(obs_radang):
    return np.digitize(obs_radang, bins) #nos dice en que bin cae cada observacion

Q = np.zeros((13,2))

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

for episode in range(1000):
    obs = env.reset()
    current_state = get_state(obs[2])
    done = False
    while not done:
        #elijo acccion
        action = epsilon_greedy_policy(current_state, Q, epsilon)
        #actualizo accion
        obs, reward, done, info = env.step(action)
        new_state = get_state(obs[2])
        #actualizo q
        update_q(current_state, action, reward, new_state)
        current_state = new_state
print("Me entrene")
print(Q)



def simulate_random_agent(number_of_samples):
    rewards = [0] * number_of_samples #inicializo lista para almacenar recompensas
    for index in range(number_of_samples):
        obs = env.reset()
        done = False
        while not done:
            policy = optimal_policy(get_state(obs[2]), Q)
            obs, reward, done, info = env.step(policy)
            rewards[index] += reward
            if done:
                env.reset()
    env.close()
    average = np.average(rewards)

    return average, rewards

print("Simulacion aleatoria 10000 muestreos")
start = time.time()
print("Valor esperado :")
print(simulate_random_agent(1000)[0])
end = time.time()
print("------- Timetaken(seconds) ----------------")
print(end-start)



# q_values = np.zeros((6,6,2))

# def get_angle_bucket(obs):
#     return np.floor(obs[2]*10 + 3).astype(int)

# def get_angularvelocity_bucket(obs):
#     if obs[3] < -3:
#         return 0
#     elif obs[3] < -1.5:
#         return 1
#     elif obs[3] < 0:
#         return 2
#     elif obs[3] < 1.5:
#         return 3
#     elif obs[3] < 3:
#         return 4
#     else:
#         return 5

# def get_bucket_coordinates(obs):
#    return get_angle_bucket(obs), get_angularvelocity_bucket(obs)

# # defino un algoritmo epsilon greedy que me dice la proxima accion a tomar
# def get_next_action(angle_bucket, angular_velocity_bucket, epsilon):
#     if np.random.random() < epsilon:
#         return np.argmax(q_values[angle_bucket, angular_velocity_bucket])
#     else:
#         return np.random.randint(0,1)

# # #defino params de entrenamiento
# epsilon = 0.1
# discount_factor = 1
# learning_rate = 0.1

# # # entreno
# for episode in range(10000):
#     obs = env.reset()
#     done = False
#     while not done:
#         location_angle, location_angular_velocity = get_bucket_coordinates(obs)
#         action = get_next_action(location_angle, location_angular_velocity, epsilon)
#         obs, reward, done, info = env.step(action)
#         new_location_angle, new_location_angular_velocity = get_bucket_coordinates(obs)

#         old_q_value = q_values[location_angle, location_angular_velocity, action]
#         temporal_difference = reward + (discount_factor * np.max(q_values[new_location_angle, new_location_angular_velocity])) - old_q_value

#         new_q_value = old_q_value + (learning_rate * temporal_difference)
#         q_values[location_angle, location_angular_velocity, action] = new_q_value


# print(q_values)

# #pruebo bichito entrenado
# rewards = np.zeros(1000)
# for index in range(1000):
#     obs = env.reset()
#     done = False
#     while not done:
#         location_angle, location_angular_velocity = get_bucket_coordinates(obs)
#         policy = get_next_action(location_angle, location_angular_velocity, 1)
#         obs, reward, done, info = env.step(policy)
#         rewards[index] += reward
#         if done:
#             env.reset()
#     env.close()
#     average = np.average(rewards)

# print("Simulacion q learning 1000 muestreos")
# start1 = time.time()
# print("Valor esperado :")
# print(average)
# end1 = time.time()
# print("------- Timetaken(seconds) ----------------")
# print(end1-start1)

# # # El resultado de esta simulacion da un valor esperado de 8, es decir, peor que el aleatorio
# # # se entiende que esto puede ocurrir si nuestro agente esta entrenando en funcion de unicamente la posicion, y no la velocidad, es decir, ignorando parte de las percepciones del ambiente

# # # Supongamos que queremos modelar tambien en funcion de la velocidad, otro de los datos obtenidos, pero sin complicarnos demasiado respecto al valor de velocidad obtenido
# # # en tal caso seguramente quisieramos hacer uso de buckets, es decir, el sentido del vector velocidad es negativo o positivo(izq, der)?

# # q_values_ext = np.zeros((4191,2,2)) #ahora para cada estado de angulo posible, tiene una dimension de velocidad tambien (angulo x, velocidad hacia la izq o hacia la derecha)
# # # defino un algoritmo epsilon greedy que me dice la proxima accion a tomar
# # def get_next_action_ext(location, speed, epsilon):
# #     if np.random.random() < epsilon:
# #         return np.argmax(q_values_ext[location, speed])
# #     else:
# #         return np.random.randint(0,1)

# # def get_indexes(obs):
# #     if obs[1] < 0:
# #         return np.floor((obs[2]*10000) + 2095).astype(int), 0
# #     else:
# #         return np.floor((obs[2]*10000) + 2095).astype(int), 1
# # # no modifico mis valores de aprendizaje para validar que incluyendo otro dato tengo un impacto positivo
# # # entreno
# # for episode in range(1000000):
# #     obs = env.reset()
# #     done = False
# #     while not done:
# #         location_index, speed_index = get_indexes(obs)
# #         action = get_next_action_ext(location_index, speed_index, epsilon)
# #         obs, reward, done, info = env.step(action)
# #         if (obs[2] < -.2095 or obs[2] > .2095):
# #             break
# #         new_location_index, new_speed_index = get_indexes(obs)

# #         old_q_value = q_values_ext[location_index, speed_index, action]
# #         temporal_difference = reward + (discount_factor * np.max(q_values_ext[new_location_index, new_speed_index])) - old_q_value

# #         new_q_value = old_q_value + (learning_rate * temporal_difference)
# #         q_values_ext[location_index, speed_index, action] = new_q_value

# # #pruebo bichito entrenado
# # rewards = np.zeros(10000)
# # for index in range(10000):
# #     obs = env.reset()
# #     done = False
# #     while not done:
# #         location_index, speed_index = get_indexes(obs)
# #         policy = get_next_action_ext(location_index, speed_index, 1)
# #         obs, reward, done, info = env.step(policy)
# #         if (obs[2] < -.2095 or obs[2] > .2095):
# #             break
# #         rewards[index] += reward
# #         if done:
# #             env.reset()
# #     env.close()
# #     average = np.average(rewards)

# # print("Simulacion q learning 10000 muestreos tomando en cuenta dos datos")
# # start2 = time.time()
# # print("Valor esperado :")
# # print(average)
# # end2 = time.time()
# # print("------- Timetaken(seconds) ----------------")
# # print(end2-start2)