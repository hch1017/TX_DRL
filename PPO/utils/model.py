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
    def __init__(self, state_dim,
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape,
                 eMainTaskMode, eEleScanLine_shape,
                 eAziScanRange, WeaponLaunch):
        super(ActorCritic, self).__init__()
        
        self.fc = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
                )
        
        # actor
        self.actor1 = nn.Linear(64,fStickLat_shape)
        self.actor2 = nn.Linear(64,fStickLon_shape)
        self.actor3 = nn.Linear(64,fThrottle_shape)
        self.actor4 = nn.Linear(64,fRudder_shape)
        self.actor5 = nn.Linear(64,eMainTaskMode)
        self.actor6 = nn.Linear(64,eEleScanLine_shape)
        self.actor7 = nn.Linear(64,eAziScanRange)
        self.actor8 = nn.Linear(64,WeaponLaunch)
        
        # critic
        self.critic = nn.Linear(64,1)

    def forward(self, state):
        x = self.fc(state)
        output1 = F.softmax(self.actor1(x),0)
        output2 = F.softmax(self.actor2(x),0)
        output3 = F.softmax(self.actor3(x),0)
        output4 = F.softmax(self.actor4(x),0)
        output5 = F.softmax(self.actor5(x),0)
        output6 = F.softmax(self.actor6(x),0)
        output7 = F.softmax(self.actor7(x),0)
        output8 = F.softmax(self.actor8(x),0)
        
        return output1, output2, output3, output4, output5, output6, output7, output8
    
    def act(self, state):
        action_probs = []
        action_probs.append(F.softmax(self.actor1(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor2(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor3(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor4(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor5(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor6(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor7(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor8(self.fc(state)),0))
        
        dist = []
        for i in range(len(action_probs)):
            dist.append(Categorical(action_probs[i]))
        
        action = []
        for i in range(len(action_probs)):
            action.append(dist[i].sample())
        
        for i in range(len(action_probs)):
            if i == 0:
                action_logprob = dist[i].log_prob(action[i]).unsqueeze(0)
            else:
                action_logprob = torch.cat([action_logprob, dist[i].log_prob(action[i]).unsqueeze(0)],0)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_probs = []
        action_probs.append(F.softmax(self.actor1(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor2(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor3(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor4(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor5(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor6(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor7(self.fc(state)),0))
        action_probs.append(F.softmax(self.actor8(self.fc(state)),0))
                        
        dist = []
        for i in range(len(action_probs)):
            dist.append(Categorical(action_probs[i]))
                        
        for i in range(len(action_probs)):
            if i == 0:
                action_logprobs = dist[i].log_prob(action[:,i]).unsqueeze(1)
            else:
                action_logprobs = torch.cat([action_logprobs, dist[i].log_prob(action[:,i]).unsqueeze(1)],1)
                    
                    
        for i in range(len(action_probs)):
            if i == 0:
                dist_entropy = dist[i].entropy().unsqueeze(1)
            else:
                dist_entropy = torch.cat([dist_entropy, dist[i].entropy().unsqueeze(1)],1)

        state_values = self.critic(self.fc(state))
        
        return action_logprobs, state_values, dist_entropy

    
class PPO:
    def __init__(self, state_dim,                
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape,
                 eMainTaskMode, eEleScanLine_shape,
                 eAziScanRange, WeaponLaunch, 
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, 
                   fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape,
                 eMainTaskMode, eEleScanLine_shape,
                 eAziScanRange, WeaponLaunch).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.fc.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor1.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor2.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor3.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor4.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor5.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor6.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor7.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor8.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim,
                   fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape,
                 eMainTaskMode, eEleScanLine_shape,
                 eAziScanRange, WeaponLaunch).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())    
        
        self.MseLoss = nn.MSELoss()
        
        self.loss_record = []
        
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
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
        real_action = []
        #在交互过程中，这里还需要获得每一种动作的离散化后的列表，将argmax后得到的index放入列表中采样
        #假设叫做action_list
        action_list = []
        action_list.append(np.round(np.arange(-1.0, 1.1, 0.1),1).tolist())
        action_list.append(np.round(np.arange(-1.0, 1.1, 0.1),1).tolist())
        action_list.append(np.round(np.arange(0, 1.1, 0.1),1).tolist())
        action_list.append(np.round(np.arange(-1.0, 1.1, 0.1),1).tolist())
        action_list.append(np.array([0,1]).tolist())
        action_list.append(np.array([2,4]).tolist())
        action_list.append(np.array([30,60,120]).tolist())
        action_list.append(np.array([0,1]).tolist())
        for i in range(self.action_dim):
            real_action.append(action_list[i][action.cpu().tolist()[0][i]])
        real_action

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
        rewards = rewards.unsqueeze(1)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
#             state_values = torch.squeeze(state_values)
            
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