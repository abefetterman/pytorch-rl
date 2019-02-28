import torch
import torch.nn as nn
import gym

from pytorch_rl.algos import VanillaPolicyGradient

def train(seed=None):
    env = gym.make('CartPole-v0')
    if seed:
        torch.manual_seed_all(seed)
    
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    
    hidden_dim = 32
    
    model = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,n_acts)
    )
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    algo = VanillaPolicyGradient()
    algo.init(model)
    
    n_epochs = 100
    obs = env.reset()
    for itr in range(n_epochs):
        while algo.need_data():
            action = algo.get_action(obs)
            step_result = env.step(action.item())
            algo.observe(obs, action, step_result)
            obs,_,done,_ = step_result
            if done:
                obs = env.reset()
            
        loss, info = algo.train()
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        print('itr: {3}\tloss: {0:.2f}\tep len: {1:.1f}\tep rew: {2:.2f}'.format(loss.item(), info['batch_len'], info['batch_rew'], itr))
        
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print('bye!')