import pandas as pd
import numpy as np
import random
import os, time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import argparse
import pickle
from collections import namedtuple
from collections import deque
from itertools import count

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils.replaybuffer import BasicBuffer


# 枚举量 也按照2维处理可以吗，或者用规则
# 主任务模式 m_eMainTaskMode {超视距，视距内}
# 雷达开关机 m_eRadarOnOff {关机，开机}
# 天线俯仰扫描行数 m_eEleScanLine {2行，4行}

# float 间隔5度做离散处理可以吗
# 天线方位扫描中心 m_fAziScanCenter 弧度 0-360吧
# 天线俯仰扫描中心 m_fEleScanCenter 弧度


class DQN(nn.Module):
    def __init__(self,obs_shape,
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape,
                 eMainTaskMode, eEleScanLine_shape,
                 eAziScanRange, WeaponLaunch):
        super(DQN,self).__init__()

        self.fc = nn.Sequential(
                nn.Linear(obs_shape, 256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
                )
        self.output1 = nn.Linear(64,fStickLat_shape)
        self.output2 = nn.Linear(64,fStickLon_shape)
        self.output3 = nn.Linear(64,fThrottle_shape)
        self.output4 = nn.Linear(64,fRudder_shape)
        self.output5 = nn.Linear(64,eMainTaskMode)
        self.output6 = nn.Linear(64,eEleScanLine_shape)
        self.output7 = nn.Linear(64,eAziScanRange)
        self.output8 = nn.Linear(64,WeaponLaunch)
       
  
    def forward(self, obs):
        x = self.fc(obs)
        output1 = F.softmax(self.output1(x),0)
        output2 = F.softmax(self.output2(x),0)
        output3 = F.softmax(self.output3(x),0)
        output4 = F.softmax(self.output4(x),0)
        output5 = F.softmax(self.output5(x),0)
        output6 = F.softmax(self.output6(x),0)
        output7 = F.softmax(self.output7(x),0)
        output8 = F.softmax(self.output8(x),0)
        
        return output1, output2, output3, output4, output5, output6, output7, output8
    
    
    
class DuelingDQN(nn.Module):
    def __init__(self,obs_shape,
                 fStickLat_shape, fStickLon_shape,
                 fThrottle_shape, fRudder_shape,
                 eMainTaskMode, eEleScanLine_shape,
                 eAziScanRange, WeaponLaunch):
        super(DuelingDQN,self).__init__()

        self.fc = nn.Sequential(
                nn.Linear(obs_shape, 256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
                )
        
        self.adv1 = nn.Linear(64,fStickLat_shape)
        self.adv2 = nn.Linear(64,fStickLon_shape)
        self.adv3 = nn.Linear(64,fThrottle_shape)
        self.adv4 = nn.Linear(64,fRudder_shape)
        self.adv5 = nn.Linear(64,eMainTaskMode)
        self.adv6 = nn.Linear(64,eEleScanLine_shape)
        self.adv7 = nn.Linear(64,eAziScanRange)
        self.adv8 = nn.Linear(64,WeaponLaunch)
        
        self.val1 = nn.Linear(64,1)
        self.val2 = nn.Linear(64,1)
        self.val3 = nn.Linear(64,1)
        self.val4 = nn.Linear(64,1)
        self.val5 = nn.Linear(64,1)
        self.val6 = nn.Linear(64,1)
        self.val7 = nn.Linear(64,1)
        self.val8 = nn.Linear(64,1)
  
    def forward(self, obs):
        x = self.fc(obs)
        output1 = self.val1(x) + self.adv1(x) - self.adv1(x).mean()
        output2 = self.val2(x) + self.adv2(x) - self.adv2(x).mean()
        output3 = self.val3(x) + self.adv3(x) - self.adv3(x).mean()
        output4 = self.val4(x) + self.adv4(x) - self.adv4(x).mean()
        output5 = self.val5(x) + self.adv5(x) - self.adv5(x).mean()
        output6 = self.val6(x) + self.adv6(x) - self.adv6(x).mean()
        output7 = self.val7(x) + self.adv7(x) - self.adv7(x).mean()
        output8 = self.val8(x) + self.adv8(x) - self.adv8(x).mean()
        return output1, output2, output3, output4, output5, output6, output7, output8