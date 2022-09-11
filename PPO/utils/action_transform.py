import numpy as np


# 将原始动作转化为序列数
def action_transform(action):
    action_list = []
    action_list.append(np.round(np.arange(-1.0, 1.1, 0.5), 1).tolist())
    action_list.append(np.round(np.arange(-1.0, 1.1, 0.5), 1).tolist())
    action_list.append(np.round(np.arange(0, 1.1, 0.5), 1).tolist())
    action_list.append(np.round(np.arange(-1.0, 1.1, 0.5), 1).tolist())
    action_list.append(np.array([2, 4]).tolist())
    action_list.append(np.array([30, 60, 120]).tolist())
    action_list.append(np.array([0, 1]).tolist())

    ordinal_action = []
    for i in range(len(action_list)):
        ordinal_action.append(action_list[i].index(action[i]))

    return ordinal_action


def action_restore(action):
    action_list = []
    action_list.append(np.round(np.arange(-1.0, 1.1, 0.5), 1).tolist())
    action_list.append(np.round(np.arange(-1.0, 1.1, 0.5), 1).tolist())
    action_list.append(np.round(np.arange(0, 1.1, 0.5), 1).tolist())
    action_list.append(np.round(np.arange(-1.0, 1.1, 0.5), 1).tolist())
    action_list.append(np.array([2, 4]).tolist())
    action_list.append(np.array([30, 60, 120]).tolist())
    action_list.append(np.array([0, 1]).tolist())

    real_action = []
    for i in range(self.action_dim):
        real_action.append(action_list[i][action.cpu().tolist()[0][i]])
    real_action

    return 