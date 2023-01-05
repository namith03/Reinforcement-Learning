
import torch
from utils import to_device
import numpy as np
import math

# # Equation 1
def estimate_returns(rewards, masks, gamma, device):
    rewards, masks = to_device(torch.device('cpu'), rewards, masks)
    tensor_type = type(rewards)
    output = []
    i=0
    tempsum=0
    for step in range(rewards.shape[0]):
        tempsum+= int((gamma**i)*rewards[step])
        #temp1.append(1)
        if masks[step]:
            i+=1

        else:
            t = np.ones(i+1)
            output.append(t*tempsum)
            i=0
            tempsum=0
    
    
    #print(len(output))
    returns = torch.tensor(output.copy())
    
    returns = torch.reshape(returns, (rewards.shape[0],))
    #print(returns.shape)
    return returns

# Equation 2
def estimate_rtg(rewards, masks, gamma, device):
    rewards, masks = to_device(torch.device('cpu'), rewards, masks)
    tensor_type = type(rewards)
    rtg = torch.zeros((rewards.shape[0]))
    interrtg = torch.zeros((rewards.shape[0]))
    output = []
    temp1 = []
    subsum = []
    i=0
    for step in range(rewards.shape[0]):
        subsum.append((gamma**i)*rewards[step])
        if masks[step]:
            i+=1
        else:
            subsum.reverse()
            temp1 = np.cumsum(subsum)
            temp1 = np.flip(temp1)
            
            output.append(temp1)
            subsum=[]
            temp1=[]
            i=0
            
    interrtg = torch.tensor(output.copy())
    interrtg = torch.reshape(interrtg, (rewards.shape[0],))
    
    return interrtg
    
def estimate_net_grad(rewards, masks,states,actions,gamma,theta,device):
    # these computations would be performed on CPU
    # rewards, masks, values = to_device(torch.device('cpu'), rewards, masks)
    # tensor_type = type(rewards)
    one = torch.ones((states.shape[0],1))

    states = torch.cat((states, one), dim=1)
    
    states = states.T
    actions = actions.T
    
    #print(actions.shape)
    #print(states.shape)
    #print(theta.shape)
    #print(torch.matmul(states.T, theta).shape)
    #rtg = estimate_rtg(rewards, masks, gamma, device)
    rtg = estimate_returns(rewards, masks, gamma, device)
    
    rtg = (rtg - rtg.mean()) / rtg.std()
    xyz = (actions.T - torch.matmul(states.T, theta))
    #print(xyz.shape)
    diagrtg = torch.diag(rtg)
    #print(rtg.shape)
    #print(diagrtg.shape)
    del_log_pi = torch.matmul(states,diagrtg)
    #print(del_log_pi.shape)
    grad = torch.matmul(del_log_pi,xyz)
    #print(torch.shape(rtg))
    #print(torch.shape(del_log_pi))
    
    # standardize returns for algorithmic stability
    #returns = (returns - returns.mean()) / returns.std()

    grad = grad / (torch.norm(grad) + 1e-8)

    #returns = to_device(device, grad)
    return grad

