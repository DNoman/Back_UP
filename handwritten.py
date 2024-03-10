import flowersdata as data
import random
import math

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t / total for t in temp]

def log_loss(activations,targets):
    losses = [-t * math.log(a) - (1 - t) * math.log(1-a) for a,t in zip(activations,targets)]
    return sum(losses)

epochs = 1
batch_size = 400
learning_rate = 0.4
input_count, hidden_count, output_count = 2,4,3
w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
w_h_o = [[random.random() -0.5 for _ in range(hidden_count) ] for _ in range (output_count)]
b_i_h = [0 for _ in range(hidden_count)]
b_h_o = [0 for _ in range(output_count)]
# print(len(w_i_h)) # 20
# print(len(w_i_h[0])) # 784
# print(len(w_h_o)) #10
# print(len(w_h_o[0])) #20
# print(len(b_i_h)) #20
# print(len(b_h_o)) #10

#train the network
for epoch in range(epochs):
    # for labels,targets,inputs in reader.get_training_samples(batch_size):
    pred_h = [[sum([w * a for w,a in zip(weights, inp)]) + bias for weights,bias in zip (w_i_h,b_i_h)] for inp in data.inputs]
    act_h = [[ max(0,p) for p in pred ] for pred in pred_h]
    # act_h_o = [max(pred)  for pred in pred_h]
    # print(act_h)
    # print()
    # print(act_h_o)
    pred_o = [[sum([w * a for w,a in zip(weights,inp)]) + bias for weights,bias in zip(w_h_o,b_h_o)] for inp in act_h]
    act_o = [softmax(predictions) for predictions in pred_o]
    # act_o1 = [[p for p in predictions] for predictions in pred_o]
    # act_o2 = [predictions for predictions in pred_o]
    #print(act_o)
    # print()
    # print(act_o1)
    # print()
    # print(act_o2)
    # cost
    cost = sum([log_loss(a,t) for a,t in zip(act_o,data.targets)]) / len(act_o)
    print(f"epoch:{epoch} cost:{cost:.4f}")

    # error derivatives

    errors_d_o = [[ a - t for a,t in zip(ac,ta)] for ac, ta in zip (act_o,data.targets)]
    w_h_o_T = list(zip(*w_h_o))
    errors_d_h = [[sum([d * w for d,w in zip (deltas,weights)]) * ( 0 if p <= 0 else 1 ) for weights,p in zip(w_h_o_T,pred)] for deltas , pred in zip(errors_d_o,pred_h)]
    # print(errors_d_o)
    #gradient hidden-> output
    act_h_T = list(zip(*act_h))
    errors_d_o_T = list(zip(*errors_d_o))
    w_h_o_d = [[sum([d * a for d,a in zip(deltas,act)]) for deltas in errors_d_o_T] for act in act_h_T]
    b_h_o_d = [sum([d for d in deltas]) for deltas in errors_d_o_T]
    # print(b_h_o_d)

    # Gradient input ->hidden
    inputs_T = list(zip(*data.inputs))
    errors_d_h_T = list(zip(*errors_d_h))
    w_i_h_d = [[sum([d * a for d,a in zip(deltas,act)]) for deltas in errors_d_h_T] for act in inputs_T]
    b_i_h_d = [sum([d for d in deltas]) for deltas in errors_d_h_T]
    # print(w_i_h_d)

    #update weights and biases for all layers
    w_h_o_d_T = list(zip(*w_h_o_d))
    for y in range(output_count):
        for x in range(hidden_count):
            w_h_o[y][x] -= learning_rate * w_h_o_d_T[y][x] / len(data.inputs)
        b_h_o[y] -= learning_rate * b_h_o_d[y] / len(data.inputs)

    w_i_h_d_T = list(zip(*w_i_h_d))
    for y in range(hidden_count):
        for x in range(input_count):
            w_i_h[y][x] -= learning_rate * w_i_h_d_T[y][x] / len(data.inputs)
        b_i_h[y] -= learning_rate * b_i_h_d[y]/len(data.inputs)

# labels, targets, inputs = reader.get_test_samples()
pred_h = [[sum([w * a for w,a in zip(weights,inp)]) + bias for weights,bias in zip(w_i_h,b_i_h)] for inp in data.test_inputs]
act_h = [[max(0,p) for p in pred] for pred in pred_h]
pred_o = [[sum([w * a for w,a in zip(weights,act)]) + bias for weights, bias in zip(w_h_o,b_h_o)] for act in act_h]
act_o = [softmax(predictions) for predictions in pred_o]
correct = 0
for a,t in zip(act_o,data.test_targets):
    # ma_neuron = a.index(max(a))
    # ma_target = t.index(max(t))
    if a.index(max(a)) == t.index(max(t)):
        correct += 1
    # else:
    #     print(f"degit:{ma_target}, guessed:{ma_neuron}")

print(f"Correct: {correct}/{len(act_o)} ({correct/len(act_o):%})")


