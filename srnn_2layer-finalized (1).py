import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import math
import torch.nn.functional as F
from torch.utils import data

torch.manual_seed(0)

train_X = np.load('data/trainX_10ms.npy')
train_y = np.load('data/trainY_10ms.npy').astype(np.float)

test_X = np.load('data/testX_10ms.npy')
test_y = np.load('data/testY_10ms.npy').astype(np.float)

print('dataset shape: ', train_X.shape)
print('dataset shape: ', test_X.shape)

batch_size = 128

tensor_trainX = torch.Tensor(train_X)  # transform to torch tensor
tensor_trainY = torch.Tensor(train_y)
train_dataset = data.TensorDataset(tensor_trainX, tensor_trainY)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
tensor_testX = torch.Tensor(test_X)  # transform to torch tensor
tensor_testY = torch.Tensor(test_y)
test_dataset = data.TensorDataset(tensor_testX, tensor_testY)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



lens = 0.5  # hyper-parameters of approximate function
num_epochs = 50  # 150  # n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)



b_j0 = 0.01  # neural threshold baseline
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale



class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        return grad_input * temp.float() * gamma


act_fun_adp = ActFun_adp.apply



def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    alpha = torch.exp(-1. * dt / tau_m)#.cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem


class RNN_custom(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_custom, self).__init__()

        self.hidden_size = hidden_size
        # self.hidden_size = input_size
        self.i_2_h1 = nn.Linear(input_size, hidden_size[0])
        self.h1_2_h1 = nn.Linear(hidden_size[0], hidden_size[0])
        self.h1_2_h2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.h2_2_h2 = nn.Linear(hidden_size[1], hidden_size[1])

        self.h2o = nn.Linear(hidden_size[1], output_size)

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(hidden_size[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(hidden_size[1]))
        self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))
        self.tau_m_h1 = nn.Parameter(torch.Tensor(hidden_size[0]))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(hidden_size[1]))
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        nn.init.orthogonal_(self.h1_2_h1.weight)
        nn.init.orthogonal_(self.h2_2_h2.weight)
        nn.init.xavier_uniform_(self.i_2_h1.weight)
        nn.init.xavier_uniform_(self.h1_2_h2.weight)
        nn.init.xavier_uniform_(self.h2_2_h2.weight)
        nn.init.xavier_uniform_(self.h2o.weight)

        nn.init.constant_(self.i_2_h1.bias, 0)
        nn.init.constant_(self.h1_2_h2.bias, 0)
        nn.init.constant_(self.h2_2_h2.bias, 0)
        nn.init.constant_(self.h1_2_h1.bias, 0)

        nn.init.constant_(self.tau_adp_h1, 50)
        nn.init.constant_(self.tau_adp_h2, 100)
        nn.init.constant_(self.tau_adp_o, 100)
        nn.init.constant_(self.tau_m_h1, 10.)
        nn.init.constant_(self.tau_m_h2, 10.)
        nn.init.constant_(self.tau_m_o, 15.)


        self.b_h1 = self.b_h2 = self.b_o = 0

    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape
        self.b_h1 = self.b_h2 = self.b_o = b_j0

        mem_layer1 = spike_layer1 = torch.rand(batch_size, self.hidden_size[0]).cuda()
        mem_layer2 = spike_layer2 = torch.rand(batch_size, self.hidden_size[1]).cuda()
        mem_output = torch.rand(batch_size, output_dim).cuda()
        output = torch.zeros(batch_size, output_dim).cuda()

        hidden_spike_ = []
        hidden_mem_ = []
        h2o_mem_ = []

        for i in range(seq_num):
            input_x = input[:, i, :]

            h_input = self.i_2_h1(input_x.float()) + self.h1_2_h1(spike_layer1)
            mem_layer1, spike_layer1, theta_h1, self.b_h1 = mem_update_adp(h_input, mem_layer1, spike_layer1,
                                                                         self.tau_adp_h1, self.b_h1,self.tau_m_h1)
            h2_input = self.h1_2_h2(spike_layer1) + self.h2_2_h2(spike_layer2)
            mem_layer2, spike_layer2, theta_h2, self.b_h2 = mem_update_adp(h2_input, mem_layer2, spike_layer2,
                                                                         self.tau_adp_h2, self.b_h2, self.tau_m_h2)
            mem_output = output_Neuron(self.h2o(spike_layer2), mem_output, self.tau_m_o)
            if i > 0:
                output= output + F.softmax(mem_output, dim=1)

            hidden_spike_.append(spike_layer1.data.cpu().numpy())
            hidden_mem_.append(mem_layer1.data.cpu().numpy())
            h2o_mem_.append(mem_output.data.cpu().numpy())

        return output, hidden_spike_, hidden_mem_, h2o_mem_


'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 700
hidden_dim = [128,128]  # 128
output_dim = 20
seq_dim = 100  # Number of steps to unroll
num_encode = 700
total_steps = seq_dim

model = RNN_custom(input_dim, hidden_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-2  # 1e-2


base_params = [model.i_2_h1.weight, model.i_2_h1.bias,
               model.h1_2_h1.weight, model.h1_2_h1.bias,
               model.h1_2_h2.weight, model.h1_2_h2.bias,
               model.h2_2_h2.weight, model.h2_2_h2.bias,
               model.h2o.weight, model.h2o.bias]
optimizer = torch.optim.Adam([
    {'params': base_params},
    {'params': model.tau_adp_h1, 'lr': learning_rate * 5},
    {'params': model.tau_adp_h2, 'lr': learning_rate * 5},
    {'params': model.tau_adp_o, 'lr': learning_rate * 5},
    {'params': model.tau_m_h1, 'lr': learning_rate * 2},
    {'params': model.tau_m_h2, 'lr': learning_rate * 2},
    {'params': model.tau_m_o, 'lr': learning_rate * 2}],
    lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=.5)


def train(model, num_epochs=150):
    acc = []
    best_accuracy = 80
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
            labels = labels.long().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs, _,_,_ = model(images)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
        scheduler.step()
        accuracy = test(model, train_loader)
        ts_acc = test(model)
        if ts_acc > best_accuracy and accuracy > 80:
            torch.save(model, './model/model_' + str(ts_acc) + '-readout-2layer-v1-12Feb[128,128].pth')
            best_accuracy = ts_acc
        acc.append(accuracy)
        print('epoch: ', epoch, '. Loss: ', loss.item(), '. Tr Accuracy: ', accuracy, '. Ts Accuracy: ', ts_acc)
    return acc


def test(model, dataloader=test_loader):
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in dataloader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, _,_,_ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100. * correct.numpy() / total
    return accuracy




###############################
acc = train(model, num_epochs)
accuracy = test(model)
print(' Accuracy: ', accuracy)

