import os
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Normal

import matplotlib.pyplot as plt
import pickle
from collections import namedtuple

from utils.replaybuffer import RolloutBuffer
from utils.dist import MultiCategoricalDistribution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape, 
                 eEleScanLine_shape, eAziScanRange, WeaponLaunch,
                 has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.action_dims = [fStickLat_shape, fStickLon_shape, fThrottle_shape, fRudder_shape,
                           eMainTaskMode, eEleScanLine_shape, eAziScanRange, WeaponLaunch]
        self.mcd = MultiCategoricalDistribution(self.action_dims)
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 256),
                            nn.Tanh(),
                            nn.Linear(256, 128),
                            nn.Tanh(),
                            nn.Linear(128, sum(self.action_dims))
#                             nn.Softmax(dim=-1)
                        )
#             self.output1 = nn.Linear(64, fStickLat_shape)
#             self.output2 = nn.Linear(64, fStickLon_shape)
#             self.output3 = nn.Linear(64, fThrottle_shape)
#             self.output4 = nn.Linear(64, fRudder_shape)
#             self.output5 = nn.Linear(64, eMainTaskMode)
#             self.output6 = nn.Linear(64, eEleScanLine_shape)
#             self.output7 = nn.Linear(64, eAziScanRange)
#             self.output8 = nn.Linear(64, WeaponLaunch)
            
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self, state):
        self.mcd.proba_distribution(self.actor(state).unsqueeze(0))
        action = self.mcd.get_actions(deterministic=False)
        real_action = []
        #在交互过程中，这里还需要获得每一种动作的离散化后的列表，将argmax后得到的index放入列表中采样
        #假设叫做action_list
        action_list = []
        action_list.append(np.arange(-1.0, 1.1, 0.1))
        action_list.append(np.arange(-1.0, 1.1, 0.1))
        action_list.append(np.arange(0, 1.1, 0.1))
        action_list.append(np.arange(-1.0, 1.1, 0.1))
        action_list.append(np.array([0,1]))
        action_list.append(np.array([2,4]))
        action_list.append(np.array([30,60,120]))
        action_list.append(np.array([0,1]))
#         print(action)
        for i in range(self.action_dim):
            real_action.append(action_list[i][action.cpu().tolist()[0][i]])
        return torch.FloatTensor(real_action)
#         raise NotImplementedError
    
    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        else:
            self.mcd.proba_distribution(self.actor(state).unsqueeze(0))
            action = self.mcd.get_actions(deterministic=False)
            action_logprob = self.mcd.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
        else:
            self.mcd.proba_distribution(self.actor(state))
            action = self.mcd.get_actions(deterministic=False)
            logprob = self.mcd.log_prob(action)
            dist_entropy = self.mcd.entropy()
        state_values = self.critic(state)
        
        return logprob, state_values, dist_entropy

    


class PPO:
    def __init__(self, state_dim, action_dim,                
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape,
                 eMainTaskMode, eEleScanLine_shape,
                 eAziScanRange, WeaponLaunch, 
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        if self.has_continuous_action_space == False:
            self.policy = ActorCritic(state_dim, action_dim,
                                  fStickLat_shape, fStickLon_shape,
                                  fThrottle_shape, fRudder_shape,
                                  eMainTaskMode, eEleScanLine_shape,
                                  eAziScanRange, WeaponLaunch, 
                                      has_continuous_action_space, action_std_init).to(device)
            self.optimizer = torch.optim.Adam([
                            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                        ])

            self.policy_old = ActorCritic(state_dim, action_dim,
                                  fStickLat_shape, fStickLon_shape,
                                  fThrottle_shape, fRudder_shape,
                                  eMainTaskMode, eEleScanLine_shape,
                                  eAziScanRange, WeaponLaunch, 
                                  has_continuous_action_space, action_std_init).to(device)
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
            self.optimizer = torch.optim.Adam([
                            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                        ])

            self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
            self.policy_old.load_state_dict(self.policy.state_dict())    
        
        self.MseLoss = nn.MSELoss()
        
        self.loss_record = []

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        
    def get_advantages(self, values, masks, rewards, gamma):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i-1] + gamma * values[i] * masks[i-1] - values[i-1]
            gae = delta + gamma * 0.95 * masks[i-1] * gae
            returns.insert(0, gae + values[i-1])

        adv = np.array(returns) - values.detach().numpy()
        adv = torch.tensor(adv.astype(np.float32)).float()
        # Normalizing advantages
        return returns, (adv - adv.mean()) / (adv.std() + 1e-5)
    
    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
#                 state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            real_action = []
            #在交互过程中，这里还需要获得每一种动作的离散化后的列表，将argmax后得到的index放入列表中采样
            #假设叫做action_list
            action_list = []
            action_list.append(np.arange(-1.0, 1.1, 0.1))
            action_list.append(np.arange(-1.0, 1.1, 0.1))
            action_list.append(np.arange(0, 1.1, 0.1))
            action_list.append(np.arange(-1.0, 1.1, 0.1))
            action_list.append(np.array([0,1]))
            action_list.append(np.array([2,4]))
            action_list.append(np.array([30,60,120]))
            action_list.append(np.array([0,1]))
            for i in range(self.action_dim):
                real_action.append(action_list[i][action.cpu().tolist()[0][i]])
            real_action
            
#             self.buffer.states.append(state)
#             self.buffer.actions.append(action.squeeze(0))
#             self.buffer.logprobs.append(action_logprob)

            return real_action

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        if self.buffer.is_terminals[-1]:
            discounted_reward = 0
        else:
            discounted_reward = self.policy_old.critic(self.buffer.states[-1]).item()
            
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
#             if is_terminal:
#                 discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
#             print('logprobs', logprobs.shape)
#             print('old', old_logprobs.shape)
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
#             print('r',rewards.shape)
#             print('sv', state_values.shape)

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
#             print('ent',dist_entropy.shape)
#             print('su1',surr1.shape)
#             print('su2',surr2.shape)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            self.loss_record.append(loss.mean().cpu().detach().item())
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))