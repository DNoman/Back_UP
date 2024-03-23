import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

class MLP():
    def __init__(self,din,dout):
        self.w = np.random.rand(dout,din)
        self.b = np.random.rand(dout)
        self.dout = dout
        self.din = din

    def forward(self,x):
        self.x = x
        return x @ self.w.T + self.b

    def backward(self,gradout):
        jacobian_x = self.w
        jacobian_b = np.eye(self.dout)
        jacobian_w = np.zeros((self.dout,self.dout * self.din))
        for i in range(self.dout):
            jacobian_w[i,i * self.din: (i+1) * self.din] = self.x

        self.deltaw = (gradout @ jacobian_w).reshape(self.w.shape) #shape dout * din
        self.deltab = gradout @ jacobian_b


        return gradout @ jacobian_x

    def __call__(self, x):
        return self.forward(x)

    def load(self,path: str):
        self.w = np.load(path + '_W.npy')
        self.b = np.load(path + '_b.npy')

    def save(self,path: str):
        np.save(path + '_W',self.w)
        np.save(path + '_b',self.b)

class RELU():
    def forward(self,x):
        self.x = x
        return np.maximum(0,x)

    def backward(self,gradout):
        din = dout = self.x.shape[1]
        jacobian = np.zeros((din,dout))
        for i in range(din):
            if self.x[0,i] > 0 :
                jacobian[i,i] = 1

        return gradout @ jacobian

    def __call__(self, x):
        return self.forward(x)

    def load(self, path:str):
        pass #nothing to do

    def save(self, path:str):
        pass

class CompoundNN():
    def __init__(self,blocks: list):
        self.blocks = blocks

    def forward(self,x):
        for block in self.blocks:
            x = block(x)

        return x

    def backward(self,gradout):
        for block in self.blocks[::-1]:
            gradout = block.backward(gradout)
        return gradout

    def __call__(self, x):
        return self.forward(x)

    def load(self, path:str):
        for i,block in enumerate(self.blocks):
            block.load(path + f'_{i}')

    def save(self, path:str):
        for i,block in enumerate(self.blocks):
            block.save(path + f'_{i}')

class Softmax():
    def forward(selfself,x):
        return np.exp(x) / np.exp(x).sum()
    def backward(self):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def load(self, path:str):
        pass #nothing to do

    def save(self, path:str):
        pass

class MSELoss():
    def forward(self,pred,true):
        self.pred = pred
        self.true = true
        return ((pred-true)**2).mean()
    def __call__(self, pred,true):
        return self.forward(pred,true)
    def backward(self):
        din = self.pred.shape[1]
        jacobian = 2 * (self.pred - self.true) * 1/din
        return  jacobian

######Test init
# mlp1 = MLP(6,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# x = np.random.rand(1,6)
# nn = CompoundNN([mlp1,relu1,mlp2,relu2])


####test block
# print(nn(x))
# print(relu2(mlp2(relu1(mlp1(x)))))
###test load and save
# mlp1 = MLP(6,5)
# mlp2 = MLP(6,5)
# print(mlp1(x))
# print(mlp2(x))
# mlp1.save('mlp')
# mlp2.load('mlp')
# print()
# print(mlp1(x))
# print(mlp2(x))
###test load_and save block
# mlp1 = MLP(6,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# x = np.random.rand(1,6)
# nn1 = CompoundNN([mlp1,relu1,mlp2,relu2])
#
# mlp1 = MLP(6,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# nn2 = CompoundNN([mlp1,relu1,mlp2,relu2])
#
# print(nn1(x))
# print(nn2(x))
# nn1.save('nn')
# nn2.load('nn')
# print()
# print(nn1(x))
# print(nn2(x))
#######Test backward
# mlp1 = MLP(16,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# x = np.random.rand(1,16)
# nn1 = CompoundNN([mlp1,relu1,mlp2,relu2])
#
# mlp1 = MLP(16,5)
# relu1 = RELU()
# mlp2 = MLP(5,4)
# relu2 = RELU()
# nn2 = CompoundNN([mlp1,relu1,mlp2,relu2])
# nn1(x)
# gradout = np.random.rand(1,4)
# y = nn1.backward(gradout)
# print(y)
#########loss function
# loss_fct = MSELoss()
# a = loss_fct(np.array([[1,2]]),np.array([[1.4,2.5]]))
# print(a)
# b = loss_fct.backward()
# print(b)
############Training Neural
mlp1 = MLP(6,5)
relu1 = RELU()
mlp2 = MLP(5,4)
relu2 = RELU()
x = np.random.rand(1,6)
nn = CompoundNN([mlp1,relu1,mlp2,relu2])

target = np.array([[1.,2.,3.,4.]])
x = np.random.rand(1,6)
Epochs = 100
training_loss = []
for epoch in range(Epochs):
    loss_fct = MSELoss()
    #####forward pass
    prediction = nn(x)
    loss_value = loss_fct(prediction,target ) #compute the loss
    training_loss.append(loss_value)
    gradout = loss_fct.backward()
    nn.backward(gradout)

    #Update the weights
    mlp1.w = mlp1.w - mlp1.deltaw
    mlp1.b = mlp1.b - mlp1.deltab
    mlp2.w = mlp2.w - mlp2.deltaw
    mlp2.b = mlp2.b - mlp2.deltab

plt.plot(training_loss)
print(prediction)
print(target)