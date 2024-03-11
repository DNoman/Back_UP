import random
import math
import GameEnv
import pygame
# import numpy as np
import flowersdata as data
import numpy as np
import matplotlib.pyplot as plt

input_count, hidden_count, output_count = 19, 128, 5
learning_rate = 0.01
epochs = 5000


class ANN():
    def __init__(self):
        # buidling deep neural network 2X4X3 (main network)
        self.w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
        self.w_h_o = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
        self.b_i_h = [0 for _ in range(hidden_count)]
        self.b_h_o = [0 for _ in range(output_count)]

    # buidling deep neural network 2X4X3 (target network)
    # w_i_h_2 = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
    # w_h_o_2 = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
    # b_i_h_2 = [0 for _ in range(hidden_count)]
    # b_h_o_2 = [0 for _ in range(output_count)]

    # def ReLU(Z):
    #     return np.maximum(Z, 0)

    def log_loss(self, activations, targets):
        losses = [(a - t) ** 2 for a, t in zip(activations, targets)]
        return sum(losses)

    def softmax(self, predictions):
        m = max(predictions)
        temp = [math.exp(p - m) for p in predictions]
        total = sum(temp)
        return [t / total for t in temp]

    def forward(self, w_i_h, b_i_h, w_h_o, b_h_o, inputs):
        pred_h = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_i_h, b_i_h)] for inp in
                  inputs]
        act_h = [[max(0, p) for p in pred] for pred in pred_h]
        pred_o = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_h_o, b_h_o)] for inp in
                  act_h]
        # act_o = [softmax(predictions) for predictions in pred_o]
        act_o = [[p for p in predictions] for predictions in pred_o]
        return pred_h, act_h, pred_o, act_o

    def backward_prop(self, pred_h, act_h, act_o, w_h_o, inputs, targets):
        errors_d_o = [[a - t for a, t in zip(ac, ta)] for ac, ta in zip(act_o, targets)]
        w_h_o_T = list(zip(*w_h_o))
        errors_d_h = [
            [sum([d * w for d, w in zip(deltas, weights)]) * (0 if p <= 0 else 1) for weights, p in zip(w_h_o_T, pred)]
            for
            deltas, pred in zip(errors_d_o, pred_h)]
        # print(errors_d_o)
        # gradient hidden-> output
        act_h_T = list(zip(*act_h))
        errors_d_o_T = list(zip(*errors_d_o))
        w_h_o_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_o_T] for act in act_h_T]
        b_h_o_d = [sum([d for d in deltas]) for deltas in errors_d_o_T]
        # print(b_h_o_d)

        # Gradient input ->hidden
        inputs_T = list(zip(*inputs))
        errors_d_h_T = list(zip(*errors_d_h))
        w_i_h_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_h_T] for act in inputs_T]
        b_i_h_d = [sum([d for d in deltas]) for deltas in errors_d_h_T]
        return w_h_o_d, b_h_o_d, w_i_h_d, b_i_h_d

    def update_weights(self, w_i_h, b_i_h, w_h_o, b_h_o, w_i_h_d, b_i_h_d, w_h_o_d, b_h_o_d, learning_rate, input_count,
                       hidden_count, output_count, inputs):
        w_h_o_d_T = list(zip(*w_h_o_d))
        for y in range(output_count):
            for x in range(hidden_count):
                w_h_o[y][x] -= learning_rate * w_h_o_d_T[y][x] / len(inputs)
            b_h_o[y] -= learning_rate * b_h_o_d[y] / len(inputs)

        w_i_h_d_T = list(zip(*w_i_h_d))
        for y in range(hidden_count):
            for x in range(input_count):
                w_i_h[y][x] -= learning_rate * w_i_h_d_T[y][x] / len(inputs)
            b_i_h[y] -= learning_rate * b_i_h_d[y] / len(inputs)
        return w_i_h, b_i_h, w_h_o, b_h_o

    ##############LEARN FUNCTION#####################
    def predict(self, w_i_h, b_i_h, w_h_o, b_h_o, inputs):
        pred_h, act_h, pred_o, act_o = self.forward(w_i_h, b_i_h, w_h_o, b_h_o, inputs)
        return act_o

    # def get_predictions(A2):
    #     return np.argmax(A2, 0)
    #
    #
    # def get_accuracy(predictions, Y):
    #     print(predictions, Y)
    #     return np.sum(predictions == Y) / Y.size

    def fit(self, w_i_h, b_i_h, w_h_o, b_h_o, inputs, targets, learning_rate, batchsize):
        # w_i_h, b_i_h, w_h_o, b_h_o = init_params(2, 4, 3)
        for i in range(batchsize):
            pred_h, act_h, pred_o, act_o = self.forward(w_i_h, b_i_h, w_h_o, b_h_o, inputs)
            cost = sum([self.log_loss(a, t) for a, t in zip(act_o, data.targets)]) / len(act_o)
            print(f"epoch:{i} cost:{cost:.4f}")
            # print(act_o)
            w_h_o_d, b_h_o_d, w_i_h_d, b_i_h_d = self.backward_prop(pred_h, act_h, act_o, w_h_o, inputs, targets)
            w_i_h, b_i_h, w_h_o, b_h_o = self.update_weights(w_i_h, b_i_h, w_h_o, b_h_o, w_i_h_d, b_i_h_d, w_h_o_d,
                                                             b_h_o_d,
                                                             learning_rate, 2,
                                                             4, 3, inputs)
            # if i % 10 == 0:
            #     print("Iteration: ", i)
            #     predictions = get_predictions(A2)
            #     print(get_accuracy(predictions, Y))
        return w_i_h, b_i_h, w_h_o, b_h_o


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def temporal_different(Q_main, action_train, Q_target_value, terminal, reward, discount_factor, batch_size):
    TD = Q_main
    # TD[action_train] = reward + discount_factor * Q_target_value
    sp = [(discount_factor * p * t) for p, t in zip(Q_target_value, terminal)]
    q = [(r + z) for r, z in zip(reward, sp)]
    for i in range(batch_size):
        # TD[i][action_train[i]] = q[i]
        for z in range(len(action_train)):
            if (z == action_train[i]).any():
                TD[i][z] = q[i]

    return TD


### main_net, target_net, state_input, next_state_input, action_train, reward, learning_rate, discount_factor
# state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = state_input,action_train,reward,next_state_input,terminal
def learn(w_h_o, b_h_o, w_i_h, b_i_h, w_h_o_2, b_h_o_2, w_i_h_2, b_i_h_2, state_batch, next_state_input,
          main, terminal_batch, target, action_batch, reward_batch, alpha, batch_size):
    """------------------------------PREDICT MAIN------------------------------"""
    predict_value_Q_main = main.predict(w_i_h, b_i_h, w_h_o, b_h_o, state_batch)
    # print("PREDICT MAIN:        ", predict_value_Q_main)

    """------------------------------PREDICT TARGET----------------------------"""
    predict_value_Q_target = target.predict(w_i_h_2, b_i_h_2, w_h_o_2, b_h_o_2, next_state_input)
    max_Q_target = [max(predict) for predict in predict_value_Q_target]
    expect_value = temporal_different(predict_value_Q_main, action_batch, max_Q_target, terminal_batch, reward_batch,
                                      alpha, batch_size)
    # print("PREDICT TARRGET:     ", predict_value_Q_target)
    # print("TEMPORAL DIFFERENT:  ", expect_value)
    w_i_h, b_i_h, w_h_o, b_h_o = main.fit(main.w_i_h, main.b_i_h, main.w_h_o, main.b_h_o, state_batch, expect_value,
                                          alpha, batch_size)
    return w_i_h, b_i_h, w_h_o, b_h_o


def choose_action(state, main_network):
    rand = np.random.random()
    if rand < epsilon:
        action = np.random.choice(5)
    else:
        # state = state.reshape(1,19)
        # list(zip(*w_h_o_d))
        # state = list(zip(*state))
        actions = main_network.predict(main_network.w_i_h, main_network.b_i_h, main_network.w_h_o, main_network.b_h_o,state)
        # action = np.argmax(actions)
        action = actions.index(max(actions))

    return action
######Mai sua thang nay

def save_weights_to_text_file(weights, file_path):
    with open(file_path, 'w') as file:
        for weight in weights:
            file.write(str(weight) + '\n')


def load_weights_from_text_file(file_path):
    with open(file_path, 'r') as file:
        weights = [float(line.strip()) for line in file]
    return weights


batch_size = 32
TOTAL_GAMETIME = 100000000000
N_EPISODES = 10000
REPLACE_TARGET = 50

epsilon = 1.0
epsilon_dec = 0.1
epsilon_end = 0.01
game = GameEnv.RacingEnv()
game.fps = 60

ddqn_scores = []
eps_history = []
max_steps_history = []
avg_steps_history = []
maxscore = 0

main_network = ANN()
target_network = ANN()
memory = ReplayBuffer(25000, 19, 5)


def run():
    global maxscore, epsilon
    for e in range(N_EPISODES):
        game.reset()  # reset env

        score = 0
        counter = 0

        observation_, reward, done = game.step(0)
        observation = []
        for item in range(len(observation_)):
            observation = [ [item] for _ in range(19)]
            observation = list(zip(*observation))
        # observation = [pre for pre in observation_]
        print(observation)
        gtime = 0  # set game time back to 0

        renderFlag = True  # if you want to render every episode set to true

        # if e % 10 == 0 and e > 0: # render every 10 episodes
        #     renderFlag = True

        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action = choose_action(observation, main_network)
            # print(observation)
            observation_, reward, done = game.step(action)
            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward
            observation = np.array(observation)
            memory.store_transition(observation, action, reward, observation_, int(done))
            if memory.mem_cntr > batch_size:
                state, action, reward, new_state, done_ = memory.sample_buffer(batch_size)
                learn(main_network.w_h_o, main_network.b_h_o, main_network.w_i_h, main_network.b_i_h, target_network.w_h_o,
                      target_network.b_h_o, target_network.w_i_h, target_network.b_i_h, state,
                      new_state,
                      main_network, done_, target_network, action, reward,
                      learning_rate, batch_size)

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        if epsilon > epsilon_end:
            epsilon = epsilon * epsilon_dec
        if epsilon > epsilon_end:
            epsilon = epsilon_end

        if score > 110 and score > maxscore:
            maxscore = score
            #####save weights main_network
            save_weights_to_text_file(main_network.w_i_h, 'main_network.w_i_h.weights_max.txt')
            save_weights_to_text_file(main_network.b_i_h, 'main_network.b_i_h.weights_max.txt')
            save_weights_to_text_file(main_network.w_h_o, 'main_network.w_h_o.weights_max.txt')
            save_weights_to_text_file(main_network.b_h_o, 'main_network.b_h_o.weights_max.txt')
            #####save weights target_network
            save_weights_to_text_file(target_network.w_i_h, 'main_network.w_i_h.weights_max.txt')
            save_weights_to_text_file(target_network.b_i_h, 'main_network.b_i_h.weights_max.txt')
            save_weights_to_text_file(target_network.w_h_o, 'main_network.w_h_o.weights_max.txt')
            save_weights_to_text_file(target_network.b_h_o, 'main_network.b_h_o.weights_max.txt')
            #######
        eps_history.append(e)
        ddqn_scores.append(score)
        max_steps_history.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])
        if len(max_steps_history) >= 100:
            avg_steps = np.mean(max_steps_history[-100:])
            avg_steps_history.append(avg_steps)

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            target_network.b_h_o = main_network.b_h_o
            target_network.w_h_o = main_network.w_h_o
            target_network.b_i_h = main_network.b_i_h
            target_network.w_i_h = main_network.w_i_h

        if e % 40 == 0 and e > 10:
            #####save weights main_network
            save_weights_to_text_file(main_network.w_i_h, 'main_network.w_i_h.weights.txt')
            save_weights_to_text_file(main_network.b_i_h, 'main_network.b_i_h.weights.txt')
            save_weights_to_text_file(main_network.w_h_o, 'main_network.w_h_o.weights.txt')
            save_weights_to_text_file(main_network.b_h_o, 'main_network.b_h_o.weights.txt')
            #####save weights target_network
            save_weights_to_text_file(target_network.w_i_h, 'main_network.w_i_h.weights.txt')
            save_weights_to_text_file(target_network.b_i_h, 'main_network.b_i_h.weights.txt')
            save_weights_to_text_file(target_network.w_h_o, 'main_network.w_h_o.weights.txt')
            save_weights_to_text_file(target_network.b_h_o, 'main_network.b_h_o.weights.txt')
            print("save model")

        print('episode: ', e, 'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              ' epsolon: ', epsilon,
              ' memory size', memory.mem_cntr % memory.mem_size)

        # Vẽ biểu đồ sau mỗi episode
        plt.plot(eps_history, ddqn_scores, marker='o', linestyle='-')
        if e >= 100:
            plt.plot(eps_history[100:], avg_steps_history[1:], label='Avg step', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('BIỂU ĐỒ REWARD ĐẠT ĐƯỢC TRÊN MỖI EPISODE ')
        plt.legend(['Reward', 'Avg_step'])
        plt.grid(True)
        plt.pause(0.05)  # Tạo độ trễ ngắn để cập nhật biểu đồ
        # if e % 1000 ==0:
        #     plt.show()
        #     break
        #     plt.savefig(f'plot_{e + 1}_episodes.png')
        # plt.clf()  # Xóa biểu đồ sau mỗi lần vẽ


run()
# a = ANN()
# b = ANN()
# a.fit(a.w_i_h, a.b_i_h, a.w_h_o, a.b_h_o, data.inputs, data.targets, learning_rate, epochs)
# pred_h, act_h, pred_o, act_o = a.forward(a.w_i_h, a.b_i_h, a.w_h_o, a.b_h_o, data.test_inputs)
#
# # w1, b1, w2, b2 = fit(w_i_h, b_i_h, w_h_o, b_h_o, data.inputs, data.targets, learning_rate, epochs)
# #
# # # pred_h = [[sum([w * a for w,a in zip(weights,inp)]) + bias for weights,bias in zip(w1,b1)] for inp in data.test_inputs]
# # # act_h = [[max(0,p) for p in pred] for pred in pred_h]
# # # pred_o = [[sum([w * a for w,a in zip(weights,act)]) + bias for weights, bias in zip(w2,b2)] for act in act_h]
# # # act_o = [softmax(predictions) for predictions in pred_o]
# # pred_h, act_h, pred_o, act_o = forward(w_i_h, b_i_h, w_h_o, b_h_o, data.test_inputs)
# correct = 0
# for a, t in zip(act_o, data.test_targets):
#     # ma_neuron = a.index(max(a))
#     # ma_target = t.index(max(t))
#     if a.index(max(a)) == t.index(max(t)):
#         correct += 1
#     # else:
#     #     print(f"degit:{ma_target}, guessed:{ma_neuron}")
#
# print(f"Correct: {correct}/{len(act_o)} ({correct / len(act_o):%})")
