# UCMEC-mmWave-Fronthaul
Simulation code of our paper ''Towards Decentralized Task Offloading and Resource Allocation in User-Centric Mobile Edge Computing''. This is the extended version of our previous paper ``When the User-Centric Network Meets Mobile Edge Computing: Challenges and Optimization''.

**This manuscript is currently in the review process**

We have designed a large number of portable multi-agent deep reinforcement learning (MADRL) environments of MEC to verify different algorithms. Specifically, The IPPO and MAPPO algorithms are modified based on [light-mappo](https://github.com/tinyzqh/light_mappo).

## Installation

Simply download the code, create a Conda environment, and then run the code, adding packages as needed.

##  How to use

You can quickly design your multi-agent environment using the following template. Please refer to /envs/env_core.py file.

```python
import numpy as np
class EnvCore(object):
    """
    # Environment Agent
    """
    def __init__(self):
        self.agent_num = 2 # set the number of agents(aircrafts), here set to two
        self.obs_dim = 14 # set the observation dimension of agents
        self.action_dim = 5 # set the action dimension of agents, here set to a five-dimensional

    def reset(self):
        """
        # When self.agent_num is set to 2 agents, the return value is a list, and each list contains observation data of shape = (self.obs_dim,)
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # When self.agent_num is set to 2 agents, the input of actions is a two-dimensional list, and each list contains action data of shape = (self.action_dim,).
        # By default, the input is a list containing two elements, because the action dimension is 5, so each element has a shape of (5,)
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
```

Just write this part of the code, and you can seamlessly connect with MAPPO. After env_core.py, two files, env_discrete.py and env_continuous.py, were separately extracted to encapsulate the action space and discrete action space. In elif self.continuous_action: in algorithms/utils/act.py, this judgment logic is also used to handle continuous action spaces. The # TODO here in runner/shared/env_runner.py is also used to handle continuous action spaces.

In the train.py file, choose to comment out continuous environment or discrete environment to switch the demo environment.

Other MADRL algorithms are simulated based on [epymarl](https://github.com/uoe-agents/epymarl). Thanks very much for the contributions of the authors of these repositories. Please visit the homepage of epymarl to view specific usage tutorials.
