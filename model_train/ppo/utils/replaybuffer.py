from utils.action_transform import action_transform
import torch

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
def BufferGeneration(state):
    state = state.tolist()

    # 飞机，雷达探测，光电探测，火控
    state_tmp = state[2:32] + state[34:44] + state[123:126] + state[234:241]
    
    # 电子战告
    i = 243
    for _ in range(8):
        state_tmp = state_tmp + state[i:i+4]
        i += 5
        
    # DAS警报
    i = 284
    for _ in range(8):
        state_tmp = state_tmp + state[i:i+2]
        i += 3
        
    # 低频率预警
    i = 309
    for _ in range(8):
        state_tmp = state_tmp + state[i:i+7]
        i += 8

    # 导弹回传
    # i = 374
    # for _ in range(16):
    #     state_tmp = state_tmp + state[i:i+8]
    #     i += 9

    # 额外特征
    state_tmp = state_tmp + state[428:430]
    
    action = action_transform(state[611:615] + state[616:619])
    
    action_logprob = torch.log(torch.tensor(state[619:]))
     
    return state_tmp, action, action_logprob
