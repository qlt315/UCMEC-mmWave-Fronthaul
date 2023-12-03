import pandas as pd
import scipy.io as scio
import numpy as np
import hdf5storage

# import data
data = hdf5storage.loadmat('./performance_data.mat')
reward_list = np.array(data['reward_list'])  # 将matlab数据赋值给python变量
reward_list_processed = reward_list
# data processing

rolling_intv = 35  # MADDPG: 40
df = pd.DataFrame(reward_list[6, :])
reward_list_processed[6, :] = list(np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))

rolling_intv = 40  # IQL:40
df = pd.DataFrame(reward_list[7, :])
reward_list_processed[7, :] = list(np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))

# data output
file_name = 'reward_MADDPG_coop.mat'
scio.savemat(file_name, {'reward_MADDPG_coop': reward_list_processed[6, :]})

# data output
file_name = 'reward_IQL_noncoop.mat'
scio.savemat(file_name, {'reward_IQL_noncoop': reward_list_processed[7, :]})
