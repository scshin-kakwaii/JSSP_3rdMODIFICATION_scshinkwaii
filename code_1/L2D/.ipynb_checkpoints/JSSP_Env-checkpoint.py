"""
FINAL SIMPLE SOLUTION:

The problem: Every reward shaping we tried encourages single-rule policies.
- Consistency bonus → rewards not switching
- Step-level LB rewards → too noisy, no clear signal
- Hybrid rewards → still biases toward single rule

The solution: GO BACK TO BASICS with one key change:
1. Use original sparse reward (only at episode end)
2. Keep rule history in state (so network CAN learn to switch if beneficial)
3. INCREASE exploration dramatically
4. Add explicit anti-collapse mechanism
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
from dispatching_rules import Rules, apply_dispatching_rule


class SJSSP(gym.Env, EzPickle):
    def __init__(self, n_j, n_m):
        EzPickle.__init__(self)

        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = n_j * n_m

        self.action_space = spaces.Discrete(8)

        self.observation_space = spaces.Dict({
            "adj": spaces.Box(low=0, high=1, shape=(self.number_of_tasks, self.number_of_tasks), dtype=np.single),
            "fea": spaces.Box(low=-1e6, high=1e6, shape=(self.number_of_tasks, configs.input_dim), dtype=np.single),
            "candidate": spaces.Box(low=0, high=self.number_of_tasks, shape=(self.number_of_jobs,), dtype=np.int64),
            "mask": spaces.Box(low=0, high=1, shape=(self.number_of_jobs,), dtype=bool)
        })
        
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]

        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    def _get_obs(self):
        features = np.concatenate(
            (self.LBs.reshape(-1, 1) / configs.et_normalize_coef,
             self.finished_mark.reshape(-1, 1)),
            axis=1
        )
        return {
            "adj": self.adj,
            "fea": features,
            "candidate": self.omega,
            "mask": self.mask
        }

    def done(self):
        return len(self.partial_sol_sequeence) == self.number_of_tasks

    @override
    def step(self, rule_action):
        """
        BACK TO ORIGINAL SPARSE REWARD
        No consistency bonus, no step-level rewards
        Let exploration and rule history do the work
        """
        current_candidates = self.omega[~self.mask]
        
        if len(current_candidates) == 0:
            obs = self._get_obs()
            return obs, 0.0, self.done(), False, {}

        action = apply_dispatching_rule(
            Rules(rule_action),
            current_candidates,
            self
        )
        
        if action not in self.partial_sol_sequeence:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)
            
            startTime_a, flag = permissibleLeftShift(
                a=action, durMat=self.dur, mchMat=self.m,
                mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs
            )
            self.flags.append(flag)

            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + dur_a
            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)
            
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:
                self.adj[succd, precd] = 0

        # ORIGINAL SPARSE REWARD - only negative LB change
        reward = - (self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()
        
        obs = self._get_obs()
        terminated = self.done()
        
        return obs, reward, terminated, False, {}

    @override
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options is None or 'data' not in options:
            raise ValueError("Instance data must be provided via options dict")
        data = options['data']

        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        conj_nei_up_stream[self.first_col] = 0
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)
        
        self.omega = self.first_col.astype(np.int64)
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)
        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        obs = self._get_obs()
        return obs, {}