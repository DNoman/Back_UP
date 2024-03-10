import numpy as np
import random
from collections import deque
import gym  # Only for the environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


class NeuralNetwork:
    def __init__(self, state_size, action_size, hidden_size=24):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # Initialize weights
        self.weights = {
            # 'w1': np.random.randn(state_size, hidden_size) * 0.5
            'w1': [[random.random() - 0.5 for _ in range(state_size)] for _ in range(hidden_size)],
            'w2': [[random.random() -0.5 for _ in range(hidden_size) ] for _ in range (action_size)]
        }

    def forward(self, state):
        self.z1 = np.dot(state, self.weights['w1'])
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.weights['w2'])
        return self.z2  # This is the Q-value for each action

    def backward(self, state, action, target, learning_rate=0.001):
        output = self.forward(state)
        delta_output = output
        delta_output[0, action] = target - output[0, action]

        delta_hidden = np.dot(delta_output, self.weights['w2'].T) * (1 - np.tanh(self.z1) ** 2)

        # Gradient descent
        dw2 = np.dot(self.a1.T, delta_output)
        dw1 = np.dot(state.T, delta_hidden)

        self.weights['w2'] += learning_rate * dw2
        self.weights['w1'] += learning_rate * dw1
class ReplayMemory:
    def __init__(self, capacity=1000):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(capacity=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.policy_net = NeuralNetwork(state_size, action_size)
        self.target_net = NeuralNetwork(state_size, action_size)
        self.update_target_network()

    def update_target_network(self):
        self.target_net.weights = self.policy_net.weights.copy()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.policy_net.forward(state)
        return np.argmax(act_values[0])  # Returns action

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_net.forward(next_state)[0]))
            target_f = self.policy_net.forward(state)
            target_f[0][action] = target
            self.policy_net.backward(state, action, target_f[0][action], self.learning_rate)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DDQNAgent(state_size, action_size)
episodes = 1000

for e in range(episodes):
    state = env.reset()

    for time in range(500):  # 500 timesteps max
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.memory.add(state, action, reward, next_state, done)
        state = next_state

        if done:
            agent.update_target_network()
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break

        if len(agent.memory.memory) > 32:
            agent.replay(32)