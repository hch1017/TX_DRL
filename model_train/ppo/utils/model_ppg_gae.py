import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from utils.replaybuffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim,
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape, eEleScanLine_shape,
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
        self.actor5 = nn.Linear(64,eEleScanLine_shape)
        self.actor6 = nn.Linear(64,eAziScanRange)
        self.actor7 = nn.Linear(64,WeaponLaunch)

        self.softmax = nn.Softmax(1)
        
        self.actor_v = nn.Linear(64,1)
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        
    def forward(self, state):
        x = self.fc(state)
        output1 = self.softmax(self.actor1(x))
        output2 = self.softmax(self.actor2(x))
        output3 = self.softmax(self.actor3(x))
        output4 = self.softmax(self.actor4(x))
        output5 = self.softmax(self.actor5(x))
        output6 = self.softmax(self.actor6(x))
        output7 = self.softmax(self.actor7(x))
        
        return output1, output2, output3, output4, output5, output6, output7
    
    def act(self, state):
        action_probs = []
        x = self.fc(state)
        action_probs.append(self.softmax((self.actor1(x))))
        action_probs.append(self.softmax((self.actor2(x))))
        action_probs.append(self.softmax((self.actor3(x))))
        action_probs.append(self.softmax((self.actor4(x))))
        action_probs.append(self.softmax((self.actor5(x))))
        action_probs.append(self.softmax((self.actor6(x))))
        action_probs.append(self.softmax((self.actor7(x))))

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
        policy_values = []
        x = self.fc(state)
        action_probs.append(self.softmax((self.actor1(x))))
        action_probs.append(self.softmax((self.actor2(x))))
        action_probs.append(self.softmax((self.actor3(x))))
        action_probs.append(self.softmax((self.actor4(x))))
        action_probs.append(self.softmax((self.actor5(x))))
        action_probs.append(self.softmax((self.actor6(x))))
        action_probs.append(self.softmax((self.actor7(x))))

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

        state_values = self.critic(state)
        policy_values = self.actor_v(x)

        return action_logprobs, state_values, dist_entropy, action_probs, policy_values

# def clipped_value_loss(values, rewards, old_values, clip):
#     value_clipped = old_values + (values - old_values).clamp(-clip, clip)
#     value_loss_1 = (value_clipped.flatten() - rewards) ** 2
#     value_loss_2 = (values.flatten() - rewards) ** 2
#     return torch.mean(torch.max(value_loss_1, value_loss_2))
        
        
class PPG:
    def __init__(self, state_dim,                
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape,
                 eEleScanLine_shape, eAziScanRange, WeaponLaunch,
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, use_gae, gae_lambda,
                entropy_beta, aux_beta, aux_epochs):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.aux_epochs = aux_epochs
        self.entropy_beta = entropy_beta
        self.aux_beta = aux_beta
        
        self.buffer = RolloutBuffer()

        # self.policy = None
        # self.optimizer = None
        # self.old_policy = None
        
        self.MseLoss = nn.MSELoss()
        self.KL_Loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        
        self.actor_loss_record = []
        self.critic_loss_record = []
        self.returns = None
    
    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob = self.old_policy.act(state)
        real_action = []
        #在交互过程中，这里还需要获得每一种动作的离散化后的列表，将argmax后得到的index放入列表中采样
        #假设叫做action_list
        action_list = []
        action_list.append(np.round(np.arange(-1.0, 1.1, 0.5),1).tolist())
        action_list.append(np.round(np.arange(-1.0, 1.1, 0.5),1).tolist())
        action_list.append(np.round(np.arange(0, 1.1, 0.5),1).tolist())
        action_list.append(np.round(np.arange(-1.0, 1.1, 0.5),1).tolist())
        action_list.append(np.array([2,4]).tolist())
        action_list.append(np.array([30,60,120]).tolist())
        action_list.append(np.array([0,1]).tolist())
        for i in range(self.action_dim):
            real_action.append(action_list[i][action.cpu().tolist()[0][i]])
        return real_action

    def update(self):
        # Monte Carlo estimate of returns
        rewards = self.buffer.rewards
        dones = self.buffer.is_terminals
        states = self.buffer.states
        last_val = self.policy.critic(states[-1])
        
        returns = []
        advantages = []
        if not self.use_gae:
            if self.buffer.is_terminals[-1]:
                discounted_reward = 0
            else:
                discounted_reward = self.old_policy.critic(self.buffer.states[-1]).item()

            for reward, is_terminal, state in zip(reversed(rewards), reversed(dones),reversed(states)):
                discounted_reward = reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
                advantages.insert(0, discounted_reward - self.policy.critic(state))
        else:
            if self.buffer.is_terminals[-1]:
                discounted_reward = 0
            else:
                discounted_reward = self.old_policy.critic(self.buffer.states[-1]).item()

            for t in reversed(range(len(self.buffer.rewards))):
                discounted_reward = rewards[t] + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
                if t == len(self.buffer.rewards)-1:
                    td_error = discounted_reward - self.policy.critic(states[t])
                    advantages.insert(0, td_error)
                else:
                    td_error = rewards[t] + self.gamma * (1-dones[t]) * self.policy.critic(states[t+1]) - self.policy.critic(states[t])
                    advantages.insert(0,advantages[0] * self.gamma * (1-dones[t]) * self.gae_lambda + td_error)
                
                
        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        returns = returns.unsqueeze(1)
        advantages = advantages.unsqueeze(1)

        self.returns = returns
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
    
        
        batch_size = 512
        iter_num = np.round(old_states.shape[0]/batch_size+0.5)
        # Optimize policy for K epochs
        for epochs in range(self.K_epochs):
            for batch in range(int(iter_num)):
                self.optimizer.zero_grad()
                
                old_states_b = old_states[(batch*batch_size):((1+batch)*batch_size)]
                old_actions_b = old_actions[(batch*batch_size):((1+batch)*batch_size)]
                old_logprobs_b = old_logprobs[(batch*batch_size):((1+batch)*batch_size)]
                rewards_b = returns[(batch*batch_size):((1+batch)*batch_size)]
                advantages_b = advantages[(batch*batch_size):((1+batch)*batch_size)]
                # Evaluating old actions and values
                logprob, state_value, dist_entropy, _, _ = self.policy.evaluate(old_states_b, old_actions_b)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprob - old_logprobs_b.detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages_b
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_b

                # final loss of clipped objective PPO
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_beta * dist_entropy.mean()
                critic_loss = 0.5*self.MseLoss(torch.squeeze(state_value), rewards_b)
                # take gradient step
                loss = actor_loss + critic_loss
                loss.backward()
                self.optimizer.step()
                self.actor_loss_record.append(actor_loss.cpu().detach().item())
                self.critic_loss_record.append(critic_loss.cpu().detach().item())

        self.old_policy.load_state_dict(self.policy.state_dict())

    def aux_update(self):
        # aux buffer需要states, advantages, critic输出的values
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs, _, _, _, _ = self.old_policy.evaluate(old_states, old_actions)
        batch_size = 64
        iter_num = np.round(old_states.shape[0] / batch_size + 0.5)
        for _ in range(self.aux_epochs):
            for batch in range(int(iter_num)):
                self.optimizer.zero_grad()
                
                old_states_b = old_states[(batch*batch_size):((1+batch)*batch_size)]
                old_actions_b = old_actions[(batch*batch_size):((1+batch)*batch_size)]
                old_logprobs_b = old_logprobs[(batch*batch_size):((1+batch)*batch_size)]
                rewards_b = self.returns[(batch*batch_size):((1+batch)*batch_size)]
                # Evaluating old actions and values

                logprob, state_values, _, action_probs, policy_values = self.policy.evaluate(old_states_b, old_actions_b)

                aux_v_loss = self.aux_beta * self.MseLoss(torch.squeeze(policy_values), rewards_b)
                kl_loss = self.KL_Loss(logprob, old_logprobs_b.detach())
                aux_actor_loss = aux_v_loss + kl_loss
                aux_critic_loss = 0.5 * self.MseLoss(torch.squeeze(state_values), rewards_b)
                aux_loss = aux_actor_loss + aux_critic_loss

                aux_loss.backward()
                self.optimizer.step()
                self.actor_loss_record.append(aux_actor_loss.cpu().detach().item())
                self.critic_loss_record.append(aux_critic_loss.cpu().detach().item())
        

    
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.old_policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage),strict=False)
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage),strict=False)
