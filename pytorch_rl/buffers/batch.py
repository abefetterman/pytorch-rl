import torch
import numpy as np
from pytorch_rl.utils import rewards_to_go

class BatchBuffer(object):
    def __init__(self):
        self.reset()
        
    def __len__(self):
        return len(self.batch_buff)
        
    def add(self, obs, action, reward, done):
        self.ep_buff.append((obs,action,done))
        self.ep_rew_buff.append(reward)
        if done:
            self.batch_buff.extend(self.ep_buff)
            self.batch_rew_buff.extend(rewards_to_go(self.ep_rew_buff))
            self.batch_lens.append(len(self.ep_rew_buff))
            self.batch_rewards.append(sum(self.ep_rew_buff))
            self.ep_buff, self.ep_rew_buff = [], []
        
    def data(self):
        obss, actions, dones = [list(x) for x in zip(*self.batch_buff)]
        rews = self.batch_rew_buff
        obss = torch.from_numpy(np.vstack(obss)).float()
        actions = torch.tensor(actions).long().unsqueeze(1)
        dones = torch.tensor(dones).int().unsqueeze(1)
        rews = torch.tensor(rews).float().unsqueeze(1)
        return [obss,actions,rews,dones]
        
    def statistics(self):
        l = len(self.batch_rewards)
        if l > 0:
            return sum(self.batch_lens)/l, sum(self.batch_rewards)/l
        return 0,0
        
    def reset(self):
        self.ep_buff, self.ep_rew_buff = [], []
        self.batch_buff, self.batch_rew_buff = [], []
        self.batch_lens, self.batch_rewards = [], []
