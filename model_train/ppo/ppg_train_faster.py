import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_ppg import PPG
from utils.AI_Interface import *
from utils.reward import *
from torch.distributions import Categorical
from utils.action_transform import action_transform
from utils.replaybuffer import BufferGeneration
import pyautogui

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(tagCenterX, tagCenterY):
    state_dim = 286  # 不带id和timestamp
    action_dim = 8
    lr_actor = 0.00003
    lr_critic = 0.0001
    gamma = 0.99
    # K_epochs = 1
    K_epochs = 50
    eps_clip = 0.2
    use_gae = True
    gae_lambda = 0.95
    aux_epochs = 50
    entropy_beta = 0.01
    aux_beta = 1

    fStickLat_shape = 5
    fStickLon_shape = 5
    fThrottle_shape = 3
    fRudder_shape = 5
    eEleScanLine_shape = 2
    eAziScanRange = 3
    WeaponLaunch = 2

    action_dims = [fStickLat_shape, fStickLon_shape, fThrottle_shape, fRudder_shape,
                   eEleScanLine_shape, eAziScanRange, WeaponLaunch]

    agent = PPG(state_dim,
                fStickLat_shape, fStickLon_shape,
                fThrottle_shape, fRudder_shape,
                eEleScanLine_shape, eAziScanRange, WeaponLaunch,
                lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                entropy_beta, aux_beta, aux_epochs)

    class ActorCritic(nn.Module):
        def __init__(self, state_dim,
                     fStickLat_shape, fStickLon_shape,
                     fThrottle_shape, fRudder_shape, eEleScanLine_shape,
                     eAziScanRange, WeaponLaunch):
            super(ActorCritic, self).__init__()

            self.fc = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )

            # actor
            self.actor1 = nn.Linear(64, fStickLat_shape)
            self.actor2 = nn.Linear(64, fStickLon_shape)
            self.actor3 = nn.Linear(64, fThrottle_shape)
            self.actor4 = nn.Linear(64, fRudder_shape)
            self.actor5 = nn.Linear(64, eEleScanLine_shape)
            self.actor6 = nn.Linear(64, eAziScanRange)
            self.actor7 = nn.Linear(64, WeaponLaunch)

            self.softmax = nn.Softmax(1)

            self.actor_v = nn.Linear(64, 1)

            # critic
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
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
                    action_logprob = torch.cat([action_logprob, dist[i].log_prob(action[i]).unsqueeze(0)], 0)

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
                    action_logprobs = dist[i].log_prob(action[:, i]).unsqueeze(1)
                else:
                    action_logprobs = torch.cat([action_logprobs, dist[i].log_prob(action[:, i]).unsqueeze(1)], 1)

            for i in range(len(action_probs)):
                if i == 0:
                    dist_entropy = dist[i].entropy().unsqueeze(1)
                else:
                    dist_entropy = torch.cat([dist_entropy, dist[i].entropy().unsqueeze(1)], 1)

            state_values = self.critic(state)
            policy_values = self.actor_v(x)

            return action_logprobs, state_values, dist_entropy, action_probs, policy_values

    agent.policy = ActorCritic(state_dim,
                               fStickLat_shape, fStickLon_shape,
                               fThrottle_shape, fRudder_shape, eEleScanLine_shape,
                               eAziScanRange, WeaponLaunch).to(device)

    agent.optimizer = torch.optim.Adam([
        {'params': agent.policy.fc.parameters(), 'lr': lr_actor},
        {'params': agent.policy.actor1.parameters(), 'lr': lr_actor},
        {'params': agent.policy.actor2.parameters(), 'lr': lr_actor},
        {'params': agent.policy.actor3.parameters(), 'lr': lr_actor},
        {'params': agent.policy.actor4.parameters(), 'lr': lr_actor},
        {'params': agent.policy.actor5.parameters(), 'lr': lr_actor},
        {'params': agent.policy.actor6.parameters(), 'lr': lr_actor},
        {'params': agent.policy.actor7.parameters(), 'lr': lr_actor},
        {'params': agent.policy.actor_v.parameters(), 'lr': lr_critic},
        {'params': agent.policy.critic.parameters(), 'lr': lr_critic}
    ])

    agent.old_policy = ActorCritic(state_dim,
                                   fStickLat_shape, fStickLon_shape,
                                   fThrottle_shape, fRudder_shape, eEleScanLine_shape,
                                   eAziScanRange, WeaponLaunch).to(device)

    agent.old_policy.load_state_dict(agent.policy.state_dict())

    if os.path.exists('checkpoint/ppg.pt'):
        agent.load('checkpoint/ppg.pt')
    elif os.path.exists('checkpoint/ppo.pt'):
        agent.load('checkpoint/ppo.pt')


    if os.path.exists('F:/TX_Distribution/temp_data/data'):
        1
    else:
        os.renames('F:/TX_Distribution/Deploy/WVR/save_observation/data', 'F:/TX_Distribution/temp_data/data')
    # os.mkdir('F:/TX_Distribution/Deploy/WVR/save_observation')
    data = pd.read_csv('F:/TX_Distribution/temp_data/data', header=None)
    pyautogui.click(tagCenterX, tagCenterY, button='left')

    if data[0].nunique() > 1:
        red_data = data[data[0] == data[0].unique()[0]].reset_index(drop=True)
        blue_data = data[data[0] == data[0].unique()[1]].reset_index(drop=True)
    else:
        red_data = data

    # for red data
    red_reward = []
    blue_reward = []
    store_another_state = []
    store_another_action = []
    store_another_action_logprob = []
    store_another_reward = []
    store_another_is_terminals = []
    for r, b in zip(red_data.iterrows(),blue_data.iterrows()):
        state_red, action_red, action_red_logprob = BufferGeneration(r[1])
        state_blue, action_blue, action_blue_logprob = BufferGeneration(b[1])

        input_r_cur, output_r_cur = getStateAndAction(r[1])
        input_b_cur, output_b_cur = getStateAndAction(b[1])
        input_r_next, _ = getStateAndAction(red_data.iloc[r[0] + 1])
        input_b_next, _ = getStateAndAction(blue_data.iloc[b[0] + 1])
        reward_once_red = getReward(input_r_cur, input_b_cur,
                                    output_r_cur, output_b_cur,
                                    input_r_next, input_b_next)
        reward_once_blue = getReward(input_b_cur, input_r_cur,
                                     output_b_cur, output_r_cur,
                                     input_b_next, input_r_next)

        if ((input_r_cur.m_AircraftBasicInfo.m_bAlive == 0 or
             input_r_cur.m_AircraftBasicInfo.m_fFuel <= 0 or
             input_r_cur.m_AircraftMoveInfo.m_dSelfAlt <= 0) or
                (input_b_cur.m_AircraftBasicInfo.m_bAlive == 0 or
                 input_b_cur.m_AircraftBasicInfo.m_fFuel <= 0 or
                 input_b_cur.m_AircraftMoveInfo.m_dSelfAlt <= 0)):
            for t in range(len(input_r_cur.m_AAMDataSet.m_AAMData)):
                if (input_r_cur.m_AAMDataSet.m_AAMData[t].m_eAAMState != 0) or \
                        (input_b_cur.m_AAMDataSet.m_AAMData[t].m_eAAMState != 0):
                    done = 0
                else:

                    done = 1
        else:
            done = 0
        if (r[0] == red_data.shape[0] - 2) or (b[0] == blue_data.shape[0] - 2):
            done = 1
            if red_data.shape[0] > blue_data.shape[0]:
                print('red win')
                reward_once_red = 50
                reward_once_blue = -50
            elif red_data.shape[0] < blue_data.shape[0]:
                print('blue win')
                reward_once_red = -50
                reward_once_blue = 50
            else:
                print('clear')
        red_reward.append(reward_once_red)
        blue_reward.append(reward_once_blue)


        state_red = torch.tensor(state_red).to(device)
        action_red = torch.tensor(action_red).to(device)
        reward_red = torch.tensor(np.array([reward_once_red])).to(device)
        done_red = torch.tensor([done]).to(device)

        state_blue = torch.tensor(state_blue).to(device)
        action_blue = torch.tensor(action_blue).to(device)
        reward_blue = torch.tensor(np.array([reward_once_blue])).to(device)
        done_blue = torch.tensor([done]).to(device)

        agent.buffer.states.append(state_red)
        agent.buffer.actions.append(action_red)
        agent.buffer.logprobs.append(action_red_logprob)
        agent.buffer.rewards.append(reward_red)
        agent.buffer.is_terminals.append(done_red)

        store_another_state.append(state_blue)
        store_another_action.append(action_blue)
        store_another_action_logprob.append(action_blue_logprob)
        store_another_reward.append(reward_blue)
        store_another_is_terminals.append(done_blue)

        if (done == 1) or (r[0] == red_data.shape[0] - 2) or (b[0] == blue_data.shape[0] - 2):
            print('red reward:', np.mean(red_reward))
            print('blue reward:', np.mean(blue_reward))
            break

    agent.update()
    agent.aux_update()
    agent.buffer.clear()

    agent.buffer.states.extend(store_another_state)
    agent.buffer.actions.extend(store_another_action)
    agent.buffer.logprobs.extend(store_another_action_logprob)
    agent.buffer.rewards.extend(store_another_reward)
    agent.buffer.is_terminals.extend(store_another_is_terminals)
    agent.update()
    agent.aux_update()
    agent.buffer.clear()
    agent.save('checkpoint/ppg.pt')
    try:
        os.remove('F:/TX_Distribution/temp_data/data')
    except:
        print('no file')
    torch.onnx.export(agent.policy.cpu(),
                      (torch.randn(1, state_dim)),
                      "F:\TX_Distribution\Deploy\WVR\AIPilots\Intelligame\model_back.onnx",
                      export_params=True,  # 是否保存训练好的参数在网络中
                      opset_version=10,  # ONNX算子版本
                      do_constant_folding=True,  # 是否不保存常数输出（优化选项）
                      input_names=['input0'],
                      output_names=['output0', 'output1', 'output2', 'output3', 'output4', 'output5', 'output6'])


if __name__ == '__main__':
    main(None, None)
