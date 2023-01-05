"""
Created on Mon Apr 23 17:21:27 2022

@author: sainamithgarapati
"""


import argparse
import gym
import os
import sys
import pickle
import time

from utils import *
from torch import nn
from agent import Agent

from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy

from point_env import PointEnv
from solutions.point_mass_solutions import estimate_net_grad

import matplotlib.pyplot as plt
#import torch

parser = argparse.ArgumentParser(description='Pytorch Policy Gradient')
parser.add_argument('--env-name', default="CartPole-v0", metavar='G',
                    help='name of the environment to run') #"Cartpole-v0"
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--gamma', type=float, default=1.00, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""cuda setting"""
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(args.gpu_index)


"""define actor and critic"""
if args.env_name == 'Point-v0':
    # we use only a linear policy for this environment
    theta = torch.normal(0, 0.01, size=(state_dim + 1, action_dim))
    policy_net = None
    theta = theta.to(dtype).to(device)
else:
    # we use both a policy and a critic network for this environment
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
    theta = None
    value_net = Value(state_dim)
    to_device(device, policy_net, value_net)

    # Optimizers
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)


"""create agent"""
if args.env_name == 'Point-v0':
    agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
              running_state=None, num_threads=args.num_threads)
else:
    agent = Agent(env, args.env_name, device, policy_net, theta, custom_reward=None,
                  running_state=None, num_threads=args.num_threads)

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
    #output = np.array(output)
    flat_list =[item for sublist in output for item in sublist]
    returns = torch.tensor(flat_list)
    
    returns = torch.reshape(returns, (rewards.shape[0],))
    #print(returns.shape)
    return returns

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
    
    flat_list =[item for sublist in output for item in sublist]
    interrtg = torch.tensor(flat_list)
    #interrtg = torch.tensor(output.copy())
    interrtg = torch.reshape(interrtg, (rewards.shape[0],))
    
    return interrtg

def update_params(batch, i_iter):
    
    #global theta
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    #print(actions)

    if args.env_name == 'CartPole-v0':
        # with torch.no_grad():
        # values = value_net(states)
        # advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, device)
        # act_l = -advantages * policy_net.get_log_prob(states, actions).sum()
        # policy_net.zero_grad()
        # act_l.backward()
        # optimizer_policy.step()
            
        # crt_l = -advantages.pow(2)
        # value_net.zero_grad()
        # crt_l.backward()
        # optimizer_value.step()
            
    
        # Question1
        #returns = estimate_returns(rewards, masks, args.gamma, device)
        returns = estimate_rtg(rewards, masks, args.gamma, device)
        loss = -torch.matmul(returns,policy_net.get_log_prob(x=states, actions=actions))
        #print(loss.shape)
        policy_net.zero_grad()
        loss.backward()
        optimizer_policy.step()
        
        # Question 2
        # returns = estimate_rtg(rewards, masks, args.gamma, device)
        # loss = -returns * policy_net.get_log_prob(x=states, actions=actions).sum()
        # policy_net.zero_grad()
        # loss.backward()
        # optimizer_policy.step()
        
        # Question 3
        
        
        """
        To implement CartPole on this env, you will need to implement the following functions:
        1. A function to compute returns, or reward to go, or advantages based on the question
            returns = estimate_returns(rewards, masks, args.gamma, device)
            rtg = estimate_rtg(rewards, masks, args.gamma, device)
            advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, device)
     
        """

    if args.env_name == 'Point-v0':
        """get values estimates from the trajectories"""
        net_grad = estimate_net_grad(rewards, masks,states,actions,args.gamma,theta,device)

        """update policy parameters"""
        theta += args.learning_rate * net_grad


def moving_average(a, n) :
    """
    Computes moving average for an array and a given window size
    and returns the average. 
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def main_loop():
    scores = []
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)

        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration) For cartpole,set mean action to False"""
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=False)
        t2 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_reward'], log_eval['avg_reward']))
            scores.append(log_eval['avg_reward'])
            


        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            # you will have to specify a proper directory to save files
            pickle.dump((policy_net, value_net), open(os.path.join(directory_for_saving_files, 'learned_models/{}_policy_grads.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net)

        # if log_eval['avg_reward'] > -22.0:
        #     args.render = True

        """clean up gpu memory"""
        torch.cuda.empty_cache()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Reward')
    plt.xlabel('Episode number')
    plt.show()

    plt.plot(moving_average(scores,100))
    #plt.xlabel("# of episodes")
    plt.ylabel(" Moving average of Reward" )
    plt.title("Rewards with smoothing")
    plt.show()
    

main_loop()