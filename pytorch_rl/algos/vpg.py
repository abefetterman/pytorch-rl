import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
import gym
import random
import numpy as np

from pytorch_rl.buffers import BatchBuffer

def loss_fn(probs, actions, rewards):
    one_hots = torch.zeros_like(probs)
    one_hots.scatter_(1, actions.view(-1,1), 1)
    log_probs = torch.sum(probs * one_hots, (1,))
    return torch.mean(log_probs * rewards)

def tensorfy(obs):
    return torch.from_numpy(obs).float().unsqueeze(0)

def train():
    env = gym.make('CartPole-v0')
    
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    
    hidden_dim = 32
    
    model = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,n_acts),
        nn.LogSoftmax()
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    buff_size = 5000
    buff = BatchBuffer()
    
    n_epochs = 100
    obs = env.reset()
    for itr in range(n_epochs):
        buff.reset()
        model.eval()
        while len(buff) < buff_size:
            logits = model(tensorfy(obs))
            action = Categorical(logits=logits).sample()
            obs, rew, done, _ = env.step(action.item())
            buff.add(obs,action,rew,done)
            if done:
                obs = env.reset()
        obss,actions,rewards,dones = buff.data()
        batch_len, batch_rew = buff.statistics()
        model.train()
        logits = model(obss)
        loss = loss_fn(logits, actions, rewards)
        loss.backward()
        optim.step()
        
        print('itr: {3}\tloss: {0:.2f}\tep len: {1:.1f}\tep rew: {2:.2f}'.format(loss.data, batch_len, batch_rew, itr))
        
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print('bye!')