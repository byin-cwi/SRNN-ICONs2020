import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import math
import keras
from torch.utils import data
import matplotlib.pyplot as plt
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", help="choose the task: smnist and psmnist", type=str,default="smnist")
parser.add_argument("--ec_f", help="choose the encode function: rbf, rbf-lc, poisson", type=str,default='rbf')
parser.add_argument("--dc_f", help="choose the decode function: adp-mem, adp-spike, integrator", type=str,default='adp-mem')
parser.add_argument("--batch_size", help="set the batch_size", type=int,default=200)
parser.add_argument("--encoder", help="set the number of encoder", type=int,default=80)
parser.add_argument("--num_epochs", help="set the number of epoch", type=int,default=200)
parser.add_argument("--learning_rate", help="set the learning rate", type=float,default=1e-2)
parser.add_argument("--len", help="set the length of the gaussian", type=float,default=0.5)
parser.add_argument('--network', nargs='+', type=int,default=[256,128])


def load_dataset(task='smnist'):
    if task == 'smnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    elif task == 'psmnist':
        X_train = np.load('./ps_data/ps_X_train.npy')
        X_test = np.load('./ps_data/ps_X_test.npy')
        y_train = np.load('./ps_data/Y_train.npy')
        y_test = np.load('./ps_data/Y_test.npy')
    else:
        print('only two task, -- smnist and psmnist')
        return 0
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    train_dataset = data.TensorDataset(X_train,y_train) # create train datset
    test_dataset = data.TensorDataset(X_test,y_test) # create test datset

    return train_dataset,test_dataset

'''
STEP 3a_v2: CREATE Adaptative spike MODEL CLASS
'''
b_j0 = .1#0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
lens = 0.5

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

def RBF_encode(x,num_neurons=5,eta=1.):
    if num_neurons<3:
        print('neurons number should be larger than 2')
        assert Exception
        return 0
    else:
        if len(x.shape) == 1:
            res = torch.zeros([x.shape[0],num_neurons]).cuda()
        if len(x.shape) == 2:
            res = torch.zeros([x.shape[0],x.shape[1],num_neurons]).cuda()

        # scale = 1./(num_neurons-2)
        # mus = [(2*i-2)/2*scale for i in range(num_neurons)]
        scale = 1./(num_neurons-2)
        mus = [(2*i-2)/2*scale for i in range(num_neurons)]

        sigmas = scale/eta
        for i in range(num_neurons):
            if len(x.shape) == 1:
                res[:,i] = gaussian(x,mu=mus[i],sigma=sigmas)
            if len(x.shape) == 2:
                res[:,:,i] = gaussian(x,mu=mus[i],sigma=sigmas)    
        return res


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma


act_fun_adp = ActFun_adp.apply



def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=1):
    #     tau_adp = torch.FloatTensor([tau_adp])
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    # tau_adp is tau_adaptative which is learnable # add requiregredients
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
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem

'''
STEP 3b: CREATE MODEL CLASS
'''


class RNN_custom(nn.Module):
    def __init__(self, input_size, hidden_dims, output_size, num_encode=30,EC_f='rbf',DC_f='mem'):
        super(RNN_custom, self).__init__()

        self.EC_f = EC_f
        self.DC_f = DC_f

        self.num_encoder = num_encode
        self.hidden_size = hidden_dims[0]
        self.num_decoder = hidden_dims[1]
        self.i2h = nn.Linear(self.num_encoder, self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2d = nn.Linear(self.hidden_size, self.num_decoder)
        self.d2d = nn.Linear(self.num_decoder, self.num_decoder)
        self.d2o = nn.Linear(self.num_decoder, output_size)

        self.tau_adp_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tau_adp_d = nn.Parameter(torch.Tensor(self.num_decoder))
        self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))
        self.tau_m_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tau_m_d = nn.Parameter(torch.Tensor(self.num_decoder))
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))
        
        if self.EC_f == 'rbf-lc':
            self.threshold_event = nn.Parameter(torch.tensor(0.2,requires_grad=True))
 

        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2d.weight)
        nn.init.xavier_uniform_(self.d2d.weight)
        nn.init.xavier_uniform_(self.d2o.weight)
        
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2d.bias, 0)
        nn.init.constant_(self.d2d.bias, 0)
        nn.init.constant_(self.d2o.bias, 0)

        nn.init.normal_(self.tau_adp_h, 700,25)
        nn.init.normal_(self.tau_adp_o, 700,25)
        nn.init.normal_(self.tau_adp_d, 700,25)

        #nn.init.normal_(self.tau_m_h, 20,5)
        #nn.init.normal_(self.tau_m_o, 100,5)
        #nn.init.normal_(self.tau_m_d, 15,5)

        #nn.init.normal_(self.tau_adp_h, 100,25)
        #nn.init.normal_(self.tau_adp_o, 300,25)
        #nn.init.normal_(self.tau_adp_d, 200,25)

        nn.init.normal_(self.tau_m_h, 20,5)
        nn.init.normal_(self.tau_m_o, 100,5)
        nn.init.normal_(self.tau_m_d, 15,5)
        self.b_h = self.b_o  = self.b_d  = 0

    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape
        self.b_h = self.b_o = self.b_d = b_j0
        
        hidden_mem = hidden_spike = torch.rand(batch_size, self.hidden_size).cuda()
        d2o_spike = output_sumspike = d2o_mem = torch.rand(batch_size, output_dim).cuda()
        h2d_mem = h2d_spike = torch.rand(batch_size, self.num_decoder).cuda()
        
        input = input/255.
        if self.EC_f[:3]=='rbf':
            input_RBF = RBF_encode(input.view(batch_size,seq_num).float(),self.num_encoder)
        
        for i in range(seq_num):
            if self.EC_f == 'rbf':
                input_x = input_RBF[:,i,:]
            elif self.EC_f == 'rbf-lc':
                input_x = input_RBF[:,i,:].gt(self.threshold_event).float().to(device)
            elif self.EC_f == 'Poisson':
                input_pixel_intensity = input[:, i, :]
                input_x = torch.rand(self.num_encoder, device='cuda') < input_pixel_intensity

            ####################################################################
            h_input = self.i2h(input_x.float()) + self.h2h(hidden_spike)
            
            hidden_mem, hidden_spike, theta_h, self.b_h = mem_update_adp(h_input,hidden_mem, hidden_spike, self.tau_adp_h, self.tau_m_h,self.b_h)
            d_input = self.h2d(hidden_spike) + self.d2d(h2d_spike)
            h2d_mem, h2d_spike, theta_d, self.b_d = mem_update_adp(d_input, h2d_mem, h2d_spike, self.tau_adp_d,self.tau_m_d, self.b_d)

            if self.DC_f[:3]=='adp':
                d2o_mem, d2o_spike, theta_o, self.b_o = mem_update_adp(self.d2o(h2d_spike),d2o_mem, d2o_spike, self.tau_adp_o, self.tau_m_o, self.b_o)
            elif self.DC_f == 'integrator':
                d2o_mem = output_Neuron(self.d2o(h2d_spike),d2o_mem, self.tau_m_o)
            if i >= 0: 
                if self.DC_f == 'adp-mem':
                    output_sumspike = output_sumspike + F.softmax(d2o_mem,dim=1)
                elif self.DC_f =='adp-spike':
                    output_sumspike = output_sumspike + d2o_spike
                elif self.DC_f =='integrator':
                    output_sumspike =output_sumspike+ F.softmax(d2o_mem,dim=1)

        return output_sumspike, hidden_spike



def train(model, num_epochs,train_loader,test_loader,file_name,MyFile):
    acc = []
    
    best_accuracy = 80
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device)
            labels = labels.long().to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs, _ = model(images)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
        scheduler.step()
        accuracy = test(model, train_loader)
        ts_acc = test(model,test_loader)
        if ts_acc > best_accuracy and accuracy > 80:
            torch.save(model, './model/model_' + str(ts_acc) + '_'+file_name+'-tau_adp.pth')
            best_accuracy = ts_acc
        acc.append(accuracy)
        res_str = 'epoch: '+str(epoch)+' Loss: '+ str(loss.item())+'. Tr Accuracy: '+ str(accuracy)+ '. Ts Accuracy: '+str(ts_acc)
        print(res_str)
        MyFile.write(res_str)
        MyFile.write('\n')
    return acc


def test(model, dataloader):
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in dataloader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100. * correct.numpy() / total
    return accuracy


def predict(model,test_loader):
    # Iterate through test dataset
    result = np.zeros(1)
    for images, labels in test_loader:
        images = images.view(-1, seq_dim, input_dim).to(device)

        outputs, _,_,_ = model(images)
        # _, Predicted = torch.max(outputs.data, 1)
        # result.append(Predicted.data.cpu().numpy())
        predicted_vec = outputs.data.cpu().numpy()
        Predicted = predicted_vec.argmax(axis=1)
        result = np.append(result,Predicted)
    return np.array(result[1:]).flatten()

if __name__ == '__main__':
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    task = args.task
    EC_f = args.ec_f
    DC_f = args.dc_f
    num_encode=args.encoder

    train_dataset,test_dataset = load_dataset(task)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


    input_dim = 1
    hidden_dims = args.network#[256,128]
    output_dim = 10
    seq_dim = int(784 / input_dim)  # Number of steps to unroll

    model = RNN_custom(input_dim, hidden_dims, output_dim,num_encode=num_encode,EC_f=EC_f,DC_f=DC_f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate

    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if EC_f == 'rbf-lc':
        base_params = [model.i2h.weight, model.i2h.bias, 
                model.h2h.weight, model.h2h.bias, 
                model.h2d.weight, model.h2d.bias,
                model.d2d.weight, model.d2d.bias, 
                model.d2o.weight, model.d2o.bias,model.threshold_event]
    else:
        base_params = [model.i2h.weight, model.i2h.bias, 
                model.h2h.weight, model.h2h.bias, 
                model.h2d.weight, model.h2d.bias,
                model.d2d.weight, model.d2d.bias, 
                model.d2o.weight, model.d2o.bias]

    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.tau_adp_h, 'lr': learning_rate * 2},
        {'params': model.tau_adp_d, 'lr': learning_rate * 3},
        {'params': model.tau_adp_o, 'lr': learning_rate * 2},
        {'params': model.tau_m_h, 'lr': learning_rate * 2},
        {'params': model.tau_m_d, 'lr': learning_rate * 2},
        {'params': model.tau_m_o, 'lr': learning_rate * 2},],
        lr=learning_rate)


    scheduler = StepLR(optimizer, step_size=25, gamma=.75)
    scheduler = MultiStepLR(optimizer, milestones=[25,50,100,150],gamma=0.5)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print('Time: ',dt_string)
    file_name = 'Task-'+task+'||Time-'+ dt_string+'||EC_f--'+EC_f+'||DC_f--'+DC_f+'||advanced'
    MyFile=open('./result_file/'+file_name+'.txt','w')
    MyFile.write(file_name)
    MyFile.write('\nnetwork: ['+str(hidden_dims[0])+' '+str(hidden_dims[1])+']')
    MyFile.write('\nlearning_rate: '+str(learning_rate))
    MyFile.write('\nbatch_size: '+str(batch_size))
    MyFile.write('\n\n =========== Result ======== \n')
    acc = train(model, num_epochs,train_loader,test_loader,file_name,MyFile)
    accuracy = test(model,test_loader)
    print('test Accuracy: ', accuracy)
    MyFile.write('test Accuracy: '+ str(accuracy))
    MyFile.close()

    ###################
    ##  Accuracy  curve
    ###################
    if num_epochs > 10:
        plt.plot(acc)
        plt.title('Learning Curve -- Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy: %')
        plt.show()

# python s_mnist-gpu.py --task smnist --ec_f rbf --dc_f adp-spike 
