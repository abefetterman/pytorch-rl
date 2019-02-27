import torch
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
import torch.nn.functional as F

from pytorch_rl.buffers import BatchBuffer

def loss_fn(probs, actions, rewards):
    one_hots = torch.zeros_like(probs)
    one_hots.scatter_(1, actions.view(-1,1), 1)
    log_probs = torch.sum(F.log_softmax(probs, 1) * one_hots, (1,))
    return -torch.mean(log_probs.unsqueeze(1) * rewards)

def tensorfy(obs):
    return torch.from_numpy(obs).float().unsqueeze(0)

class VanillaPolicyGradient:
    def __init__(self, buff_size=5000):
        self.buff_size = buff_size
    
    def init(self, model):
        self.buff = BatchBuffer()
        self.model = model
        
    def reset(self):
        self.buff.reset()
        
    def get_action(self, obs):
        logits = self.model(tensorfy(obs))
        action = Categorical(logits=logits).sample()
        return action
        
    def need_data(self):
        # returns whether we want more observations or should train
        return (len(self.buff) < self.buff_size)
    
    def observe(self, obs, action, step_result):
        new_obs, rew, done, _ = step_result
        
        self.buff.add(obs, action, rew, done)
        
    def train(self):
        obss,actions,rewards,dones = self.buff.data()
        logits = self.model(obss)
        loss = loss_fn(logits, actions, rewards)
        
        batch_len, batch_rew = self.buff.statistics()
        info = {
            'batch_len': batch_len,
            'batch_rew': batch_rew
        }
        
        return loss, info
        