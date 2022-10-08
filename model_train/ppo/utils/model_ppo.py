import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import Dataset,DataLoader
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

        # critic
        self.critic = nn.Linear(64, 1)

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

        state_values = self.critic(self.fc(state))

        return action_logprobs, state_values, dist_entropy

class store_dataset(Dataset):
    def __init__(self,old_states,old_actions,old_logprobs,rewards):
        self.old_states = old_states
        self.old_actions = old_actions
        self.old_logprobs = old_logprobs
        self.rewards = rewards


    def __getitem__(self, idx):
        return self.old_states[idx],self.old_actions[idx],self.old_logprobs[idx],self.rewards[idx]


    def __len__(self):
        return len(self.old_states)

class PPO:
    def __init__(self, state_dim,                
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape, eEleScanLine_shape,
                 eAziScanRange, WeaponLaunch, 
                 lr_actor, lr_critic, gamma, K_epochs, eps_clip, use_gae, gae_lambda):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        
        self.buffer = RolloutBuffer()

        self.policy = None
        self.optimizer = None

        self.old_policy = None
        
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
        rewards = []
        if self.buffer.is_terminals[-1]:
            discounted_reward = 0
        else:
            discounted_reward = self.old_policy.critic(self.old_policy.fc(self.buffer.states[-1])).item()
            
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
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
        batch_size = 512
        train_dataset = store_dataset(old_states, old_actions, old_logprobs, rewards)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        iter_num = np.round(old_states.shape[0]/batch_size+0.5)
        # Optimize policy for K epochs
        for epochs in range(self.K_epochs):
            loss_epoch = []
            for batch,(old_state,old_action,old_logprob,reward) in enumerate(dataloader):
                self.optimizer.zero_grad()
                
                # Evaluating old actions and values
                logprob, state_value, dist_entropy = self.policy.evaluate(old_state, old_action)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprob - old_logprob.detach())
    #             print('logprobs', logprobs.shape)
    #             print('old', old_logprobs.shape)
    #             print('logprob', logprob)
    #             print('state_value', state_value)

                # exit()
                advantages = reward - state_value.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
    #             print('r',rewards.shape)
    #             print('sv', state_values.shape)

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2).mean() + 0.5*self.MseLoss(torch.squeeze(state_value), torch.squeeze(reward)) - 0.01*dist_entropy.mean()
    #             print('ent',dist_entropy.shape)
    #             print('su1',surr1.shape)
    #             print('su2',surr2.shape)

                # take gradient step
                loss.backward()
                self.optimizer.step()
                self.loss_record.append(loss.cpu().detach().item())
                loss_epoch.append(loss.cpu().detach().item())
        # print ('epochs_loss:',np.mean(loss_epoch))
        # Copy new weights into old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.old_policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
