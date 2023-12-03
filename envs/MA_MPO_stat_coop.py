import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
# from stable_baselines3.common.env_checker import check_env
import cvxpy as cp


class MA_MPO_stat_coop(object):
    def __init__(self, render: bool = False):
        # Initialization
        gym.logger.set_level(40)
        np.random.seed(3)
        self.M = 30  # number of users
        self.N = 60  # number of APs
        self.varsig = 16  # number of antennas of each AP
        self.K = 3  # number of CPUs
        self.P_max = 0.1  # maximum transmit power of user / pilot power
        self.M_sim = 10  # number of users for simulation
        self.N_sim = 50  # number of APs for simulation

        # locations of users and APs
        self.locations_users = np.random.random_sample([self.M, 2]) * 900  # 2-D location of users
        self.locations_aps = np.random.random_sample([self.N, 2]) * 900  # 2-D location of APs

        # location of 3 CPUs
        self.locations_cpu = np.zeros([3, 2])
        self.locations_cpu[0, 0] = 300
        self.locations_cpu[0, 1] = 300
        self.locations_cpu[1, 0] = 600
        self.locations_cpu[1, 1] = 300
        self.locations_cpu[2, 0] = 450
        self.locations_cpu[2, 1] = 600
        # self.locations_cpu[3, 0] = 600
        # self.locations_cpu[3, 1] = 600

        # calculate distance between APs and users MxN matrix
        self.distance_matrix = np.zeros([self.M, self.N])
        self.distance_matrix_front = np.zeros([self.N, self.K])
        for i in range(self.M):
            for j in range(self.N):
                self.distance_matrix[i, j] = math.sqrt((self.locations_users[i, 0] - self.locations_aps[j, 0]) ** 2
                                                       + (self.locations_users[i, 1] - self.locations_aps[j, 1]) ** 2)

        for i in range(self.N):
            for j in range(self.K):
                self.distance_matrix_front[i, j] = math.sqrt((self.locations_aps[i, 0] - self.locations_cpu[j, 0]) ** 2
                                                             + (self.locations_aps[i, 1] - self.locations_cpu[
                    j, 1]) ** 2)

        # edge computing parameter
        # user parameter
        self.C_user = np.random.uniform(2e8, 5e8, [1, self.M])  # computing resource of users  in Hz
        self.Task_size = np.random.uniform(50000, 100000, [1, self.M])  # task size in bit
        self.Task_density = np.random.uniform(500, 1000, [1, self.M])  # task density cpu cycles per bit
        # Task_max_delay = np.random.uniform(2, 5, [1, M])  # task max delay in second
        self.cluster_size = 1  # AP cluster size

        # edge server parameter
        self.C_edge = np.random.uniform(10e9, 20e9, [self.K, 1])  # computing resource of edge server in CPU

        # access channel
        self.tau_c = 0.1  # coherence time = 100ms
        self.access_chan = np.zeros([self.M, self.N, self.varsig], dtype=complex)  # complex channel
        self.bandwidth_a = 2e6  # bandwidth of access channel
        self.kappa_1 = np.random.rand(1, self.N)  # parameter in Eq. (5)
        self.kappa_2 = np.random.rand(1, self.M)  # parameter in Eq. (5)
        self.f_carrier = 1.9e9  # carrier frequency in Hz
        self.h_ap = 15  # antenna height of AP
        self.h_user = 1.65  # antenna height of user
        # L = 46.3 + 33.9 * np.log10(f_carrier / 1000) - 13.82 * np.log10(h_ap) - (
        #        1.11 * np.log10(f_carrier / 1000) - 0.7) * h_user + 1.56 * np.log10(f_carrier / 1000) - 0.8
        self.L = 140.7
        self.d_0 = 10  # path-loss distance threshold
        self.d_1 = 50  # path-loss distance threshold
        self.PL = np.zeros([self.M, self.N])  # path-loss in dB
        self.beta = np.zeros([self.M, self.N])  # large scale fading
        self.gamma = np.zeros([self.M, self.N])
        self.sigma_s = 8  # standard deviation of shadow fading (dB)
        self.delta = 0.5  # parameter in Eq. (5)
        self.mu = np.zeros([self.M, self.N])  # shadow fading parameter
        self.h = np.zeros([self.M, self.N, self.varsig], dtype=complex)  # small scale fading
        self.noise_access = 3.9810717055349565e-21 * self.bandwidth_a  # noise of access channel -> -174 dbm/Hz

        for i in range(self.M):
            for j in range(self.N):
                # three slope path-loss model
                if self.distance_matrix[i, j] > self.d_1:
                    self.PL[i, j] = -self.L - 35 * np.log10(self.distance_matrix[i, j] / 1000)
                elif self.d_0 <= self.distance_matrix[i, j] <= self.d_1:
                    self.PL[i, j] = -self.L - 10 * np.log10(
                        (self.d_1 / 1000) ** 1.5 * (self.distance_matrix[i, j] / 1000) ** 2)
                else:
                    self.PL[i, j] = -self.L - 10 * np.log10((self.d_1 / 1000) ** 1.5 * (self.d_0 / 1000) ** 2)

                # Eq. (5) shadow fading computation
                self.mu[i, j] = math.sqrt(self.delta) * self.kappa_1[0, j] + math.sqrt(1 - self.delta) * self.kappa_2[
                    0, i]  # MxN matrix as Eq. (5)

                # Eq. (2) channel computation
                self.beta[i, j] = pow(10, self.PL[i, j] / 10) * pow(10, (self.sigma_s * self.mu[i, j]) / 10)
                for k in range(self.varsig):
                    self.h[i, j, k] = np.random.normal(loc=0, scale=0.5) + 1j * np.random.normal(loc=0, scale=0.5)
                    self.access_chan[i, j, k] = np.sqrt(self.beta[i, j]) * self.h[i, j, k]

        # fronthaul channel
        # front_chan = np.zeros([N, K])
        self.bandwidth_f = 2e9  # bandwidth of fronthaul channel
        self.epsilon = 6e-4  # blockage density
        self.p_ap = 1  # transmit power of APs (30 dBm = 1 W)
        self.alpha_los = 2.5  # path-loss exponent for LOS links
        self.alpha_nlos = 4  # path-loss exponent for NLOS links
        self.psi_los = 3  # Nakagami fading parameter for LOS links
        self.psi_nlos = 2  # Nakagami fading parameter for NLOS links
        self.noise_front = 1.380649 * 10e-23 * 290 * 9 * self.bandwidth_f  # fronthaul channel noise variance
        self.G = np.zeros([self.N, self.K])  # random antenna gain
        self.fai = math.pi / 6  # Main lobe beamwidth
        self.Gm = 63.1  # Directivity gain of main lobes
        self.Gs = 0.631  # Directivity gain of side lobes
        self.Gain = np.array(
            [self.Gs * self.Gs, self.Gm * self.Gm, self.Gm * self.Gs])  # random antenna gain in Eq. (7)
        self.Gain_pro = np.array(
            [(self.fai / (2 * math.pi)) ** 2, 2 * self.fai * (2 * math.pi - self.fai) / (2 * math.pi) ** 2,
             ((2 * math.pi - self.fai) / (2 * math.pi)) ** 2])

        self.P_los = np.zeros([self.N, self.K])  # probability of LOS links
        self.link_type = np.zeros([self.N, self.K])  # type of fronthaul links
        for i in range(self.N):
            for j in range(self.K):
                self.P_los[i, j] = np.exp(-self.epsilon * self.distance_matrix_front[i, j] / 1000)
                self.link_type[i, j] = np.random.choice([0, 1], p=[self.P_los[i, j],
                                                                   1 - self.P_los[i, j]])  # 0 for LOS, 1 for NLOS
                # if link_type[i, j] == 0:  # LOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_los)  # Nakagami channel gain
                # else:  # NLOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_nlos)  # Nakagami channel gain
                self.G[i, j] = np.random.choice(self.Gain, p=self.Gain_pro.ravel())
        # pilot assignment
        self.tau_p = self.M  # length of pilot symbol
        self.pilot_matrix = np.zeros([self.M, self.tau_p])
        for i in range(self.M):
            self.pilot_index = i
            self.pilot_matrix[i, self.pilot_index] = 1

        # MMSE channel estimation
        # receive_pilot = np.zeros([self.N, self.varsig, self.tau_p], dtype=complex)
        # access_chan_estimate = np.zeros([M, N, varsig], dtype=complex)
        self.theta = np.zeros([self.M, self.N])

        for i in range(self.M):
            for j in range(self.N):
                self.theta[i, j] = self.tau_p * self.P_max * (self.beta[i, j] ** 2) / (
                        self.tau_p * self.P_max * self.beta[i, j] + self.noise_access)

        # parameter init
        self.n_agents = self.M_sim
        self.agent_num = self.n_agents
        self.obs_dim = self.M_sim * 2  # set the observation dimension of agents
        self.action_dim = 4
        self._render = render
        # action space: [omega_1,omega_2,...,omega_K,p]  K+1 continuous vector for each agent
        # a in {0,1,2,3,4}, p in {0, 1, 2, 3, 4} (totally 5 levels (p+1)/5*100 mW)
        self.omega_last = np.zeros([self.M_sim])
        self.delay_last = np.zeros([self.M_sim, 1])
        self.action_space = spaces.Tuple(tuple([spaces.Discrete(4)] * self.n_agents))
        # state space: [r_1(t-1),r_2(t-1),...,r_M(t-1)]  1xM continuous vector. -> uplink rate
        # r in [0, 10e8]
        self.obs_low = np.array([0, 0])
        self.obs_high = np.array([2, 0.2])
        for i in range(self.M_sim - 1):
            self.obs_low = np.append(self.obs_low, np.array([0, 0]))
            self.obs_high = np.append(self.obs_high, np.array([2, 0.2]))
        # self.observation_space = spaces.MultiDiscrete([500] * self.M_sim)
        # self.observation_space = spaces.Box(low=self.r_low, high=self.r_high, shape=(self.M_sim,), dtype=np.float32)
        self.observation_space = spaces.Tuple(tuple(
            [spaces.Box(low=self.obs_low, high=self.obs_high, shape=(self.obs_dim,),
                        dtype=np.float32)] * self.n_agents))
        # self.np_random = None
        self.uplink_rate_access_b = np.zeros([self.M_sim, 1])
        self.step_num = 0
        self.cluster_matrix = self.cluster()

    def action_mapping(self, action_agent):
        # Transform the action space form MultiDiscrete to Discrete (1+4*3=13 cases)
        if action_agent[0] == 1:  # local processing
            omega_agent = 0
        elif action_agent[1] == 1:
            omega_agent = 1
        elif action_agent[2] == 1:
            omega_agent = 2
        elif action_agent[3] == 1:
            omega_agent = 3
        return omega_agent

    def cluster(self):
        # AP cluster
        cluster_matrix = np.zeros([self.M_sim, self.N_sim])  # AP serve user when value = 1, otherwise 0
        max_h_index_list = np.zeros(
            [self.M_sim, self.N_sim])  # sort access channel (large-scale fading) from high to low
        ap_index_list = np.zeros([self.M_sim, self.cluster_size])  # obtain the index of APs serve for each user
        for i in range(self.M_sim):
            for j in range(self.N_sim):
                max_h_index_list[i, :] = self.beta[i, 0:self.N_sim].argsort()
                max_h_index_list[i, :] = max_h_index_list[i, :][::-1]
                ap_index_list[i, :] = max_h_index_list[i, 0:self.cluster_size]
                for k in range(self.cluster_size):
                    cluster_matrix[i, int(ap_index_list[i, k])] = 1
        return cluster_matrix

    def uplink_rate_cal(self, p, omega):  # calculate the uplink transmit rate in Eq. (12)
        SINR_access = np.zeros([self.M_sim, 1])
        uplink_rate_access = np.zeros([self.M_sim, 1])
        SINR_access_mole = np.zeros([self.M_sim, 1])
        SINR_access_inter = np.zeros([self.M_sim, 1])
        SINR_access_noise = np.zeros([self.M_sim, 1])

        for i in range(self.M_sim):
            if omega[i] == 0:  # Local processing
                continue
            else:
                SINR_access_inter[i, 0] = 0  # Interferencecontinue
                SINR_access_mole[i, 0] = 0  # Useful symbol
                SINR_access_noise[i, 0] = 0  # Noise
                for j in range(self.N_sim):
                    if self.cluster_matrix[i, j] == 1:
                        SINR_access_mole[i, 0] = SINR_access_mole[i, 0] + self.theta[i, j]
                        SINR_access_noise[i, 0] = self.noise_access * self.theta[i, j]
                SINR_access_mole[i, 0] = (SINR_access_mole[i, 0] ** 2) * p[i] * self.varsig

                for k in range(self.M_sim):
                    if k == i or omega[k] == 0:
                        continue
                    else:
                        for j in range(self.N_sim):
                            if self.cluster_matrix[i, j] == 1:
                                SINR_access_inter[i, 0] = SINR_access_inter[i, 0] + self.theta[i, j] * self.beta[k, j] * \
                                                          p[k]
                SINR_access[i, 0] = SINR_access_mole[i, 0] / (SINR_access_inter[i, 0] + SINR_access_noise[i, 0])
                uplink_rate_access[i, 0] = self.bandwidth_a * np.log2(1 + SINR_access[i, 0])
        return uplink_rate_access

    def front_rate_cal(self, omega):
        chi = np.zeros([self.N_sim, self.K])  # whether an AP transmit symbol to a CPU or not
        SINR_front = np.zeros([self.N_sim, self.K])  # SINR in Eq. (7)
        front_rate = np.zeros([self.N_sim, self.K])  # Eq. (12)
        front_rate_user = np.zeros([self.M_sim, self.N_sim])
        I_sum = 0  # total sum of fronthaul interference
        for i in range(self.M_sim):
            if omega[i] == 0:
                continue
            CPU_id = int(omega[i] - 1)
            for j in range(self.N_sim):
                if self.cluster_matrix[i, j] == 1:  # This AP is belonged to the cluster of user i
                    chi[j, CPU_id] = 1

        for i in range(self.N_sim):
            for j in range(self.K):
                if chi[i, j] == 1:
                    if self.link_type[j, j] == 0:  # LOS link
                        I_sum = I_sum + self.p_ap * pow(self.distance_matrix_front[i, j] / 1000, -self.alpha_los)
                    else:
                        I_sum = I_sum + self.p_ap * pow(self.distance_matrix_front[i, j] / 1000, -self.alpha_nlos)
                else:
                    pass

        for i in range(self.N_sim):
            for j in range(self.K):
                if chi[i, j] == 1:
                    if self.link_type[i, j] == 0:  # LOS link
                        SINR_front_mole = self.p_ap * self.G[i, j] * pow(self.distance_matrix_front[i, j] / 1000,
                                                                         -self.alpha_los)
                    else:
                        SINR_front_mole = self.p_ap * self.G[i, j] * pow(self.distance_matrix_front[i, j] / 1000,
                                                                         self.alpha_nlos)
                    SINR_front[i, j] = SINR_front_mole / (I_sum - SINR_front_mole / self.G[i, j] + self.noise_front)
                    front_rate[i, j] = self.bandwidth_f * np.log2(1 + SINR_front[i, j])

        for i in range(self.M_sim):
            if omega[i] == 0:
                pass
            else:
                CPU_id = int(omega[i] - 1)
                for j in range(self.N_sim):
                    if self.cluster_matrix[i, j] == 1:
                        front_rate_user[i, j] = front_rate[j, CPU_id]

        return front_rate_user

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def reset(self):
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.obs_dim,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, action):
        self.step_num += 1
        # obtain and clip the action
        omega_current = np.zeros([self.M_sim])
        p_current = np.ones([self.M_sim]) * self.P_max
        print("Step Index:", self.step_num)
        for i in range(self.M_sim):
            omega_current[i] = self.action_mapping(action[i])
            # print("Chosen CPU ID:", omega_current)
            # print("Power:", p_current)

        uplink_rate_access = self.uplink_rate_cal(p_current, omega_current)
        front_rate_user = self.front_rate_cal(omega_current)
        self.uplink_rate_access_b = uplink_rate_access
        # print("Fronthaul Rate", front_rate_user)
        # print("Uplink Rate (Mbps):", uplink_rate_access / 10e6)
        # print("Average Uplink Rate (Mbps):", np.sum(uplink_rate_access) / (np.count_nonzero(omega_current) * 10e6))

        # local computing delay
        local_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] == 0:
                local_delay[i, 0] = self.Task_density[0, i] * self.Task_size[0, i] / self.C_user[0, i]

        # uplink delay
        uplink_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                uplink_delay_list = np.zeros([self.cluster_size, 1])
                for j in range(self.cluster_size):
                    uplink_delay_list[j, 0] = self.Task_size[0, i] / uplink_rate_access[i, 0]
                uplink_delay[i, 0] = np.max(uplink_delay_list)

        # fronthaul delay
        front_delay_matrix = np.zeros([self.M_sim, self.cluster_size])
        front_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                for j in range(self.N_sim):
                    for k in range(self.cluster_size):
                        if self.cluster_matrix[i, j] == 1:
                            front_delay_matrix[i, k] = self.Task_size[0, i] / front_rate_user[i, j]
                front_delay[i, 0] = np.max(front_delay_matrix[i, :])

        # processing delay calculation
        # solve convex problem according to Eq. (24)
        task_mat = np.zeros([self.M_sim, self.K])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                task_mat[i, CPU_id] = self.Task_size[0, i] * self.Task_density[0, i]

        # Each CPU solves a resource allocation optimization problem
        actual_C = np.zeros([self.M_sim, 1])
        for i in range(self.K):
            serve_user_id = []
            serve_user_task = []
            _local_delay = []
            _front_delay = []
            _uplink_delay = []

            for j in range(self.M_sim):
                if task_mat[j, i] != 0:
                    serve_user_id.append(j)
                    serve_user_task.append(task_mat[j, i])
                    _local_delay.append(local_delay[j, 0])
                    _front_delay.append(front_delay[j, 0])
                    _uplink_delay.append(uplink_delay[j, 0])
            if len(serve_user_id) == 0:
                continue
            C = cp.Variable(len(serve_user_id))
            _process_delay = cp.multiply(serve_user_task, cp.inv_pos(C))
            _local_delay = np.array(_local_delay)
            _front_delay = np.array(_front_delay)
            _uplink_delay = np.array(_uplink_delay)

            func = cp.Minimize(cp.sum(cp.maximum(_local_delay, _front_delay + _uplink_delay + _process_delay)))
            cons = [0 <= C, cp.sum(C) <= self.C_edge[i, 0]]
            prob = cp.Problem(func, cons)
            prob.solve(solver=cp.SCS, verbose=False)
            for k in range(len(serve_user_id)):
                _C = C.value
                actual_C[serve_user_id[k], 0] = _C[k]

        actual_process_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                actual_process_delay[i, 0] = task_mat[i, CPU_id] / actual_C[i, 0]
        '''
        process_delay = cp.max(cp.multiply(task_mat, cp.inv_pos(C)))  # Mx1
        func = cp.Minimize(cp.sum(cp.maximum(local_delay, front_delay + uplink_delay + process_delay)))
        # func = cp.Minimize(cp.sum(cp.maximum(local_delay, process_delay)))
        cons = [0 <= C]
        for i in range(K):
            cons += [cp.sum(C[:, i]) <= C_edge[i, 0]]

        prob = cp.Problem(func, cons)
        prob.solve(solver=cp.SCS, verbose=False)
        actual_C = C.value
        actual_process_delay = np.max(task_mat / actual_C, axis=1)
        # print(actual_process_delay)
        # print(C.value)
        '''

        # reward calculation
        # print("Uplink Delay:", uplink_delay)
        # print("Local Delay:", local_delay)
        # print("Front Delay:", front_delay)
        # print("Edge Processing Delay:", actual_process_delay)
        # print("Offloading Delay:", front_delay + uplink_delay + actual_process_delay)
        total_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            total_delay[i, 0] = np.maximum(local_delay[i, 0],
                                           front_delay[i, 0] + uplink_delay[i, 0] + actual_process_delay[i, 0])
        if self.step_num > 20:
            done = [1] * self.M_sim
        else:
            done = [0] * self.M_sim

        reward = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            reward[i, 0] = -0.9 * np.sum(total_delay) / self.M_sim + 0.1 * (
                    self.tau_c - np.sum(total_delay) / self.M_sim)
        print("Average Total Delay (ms):", np.sum(total_delay) * 1000 / self.M_sim)
        print("Average Uplink Rate (Mbps):", np.sum(uplink_rate_access) / (self.M_sim * 1e6))
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        observation = np.zeros([self.agent_num, self.obs_dim])
        for i in range(self.agent_num):
            if self.step_num != 1:
                observation[i, 0] = self.omega_last[i]
                observation[i, 1] = self.delay_last[i, 0]
                for j in range(self.agent_num - 1):
                    observation[i, ((j+1) * 2):((j + 2) * 2)] = np.array(
                        [self.omega_last[i], self.delay_last[i, 0]])
            observation = np.array(observation)
            sub_agent_obs.append(observation[i, :].flatten())
            sub_agent_reward.append(reward[i])
            sub_agent_done.append(done[i])
            sub_agent_info.append({})
        self.delay_last = total_delay
        self.omega_last = omega_current
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


# if __name__ == "__main__":
#     env = MA_UCMEC_Static(render=False)
#     # check_env(env)
#     obs = env.reset()
#     n_steps = 50
#     for _ in range(n_steps):
#         # Random action
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         if np.all(done):
#             obs = env.reset()
#         # print(f"state: {obs} \n")
#         print(f"action : {action}, reward : {reward}")
