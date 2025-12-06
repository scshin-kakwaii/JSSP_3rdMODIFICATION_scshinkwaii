import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
from dispatching_rules import Rules, apply_dispatching_rule, get_job_and_op_indices

class SJSSP(gym.Env, EzPickle):
    def __init__(self, n_j, n_m):
        EzPickle.__init__(self)

        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = n_j * n_m

        self.action_space = spaces.Discrete(8)
        
        # Rule feature dim: [Job_Duration, Job_Remaining_Work, Job_Ops_Remaining, Rule_Index_OneHot]
        # 3 scalar + 8 one-hot = 11 features per rule
        self.rule_fea_dim = 3 + 8

        self.observation_space = spaces.Dict({
            "adj": spaces.Box(low=0, high=1, shape=(self.number_of_tasks, self.number_of_tasks), dtype=np.single),
            "fea": spaces.Box(low=-1e6, high=1e6, shape=(self.number_of_tasks, configs.input_dim), dtype=np.single),
            "candidate": spaces.Box(low=0, high=self.number_of_tasks, shape=(self.number_of_jobs,), dtype=np.int64),
            "mask": spaces.Box(low=0, high=1, shape=(self.number_of_jobs,), dtype=bool),
            # New: Features for each of the 8 rules
            "rule_features": spaces.Box(low=-1e6, high=1e6, shape=(8, self.rule_fea_dim), dtype=np.single)
        })

        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    def get_rule_features(self):
        """
        LOGIC FIX 1: Look-ahead.
        Tính toán features cho mỗi rule TRƯỚC khi mạng neural quyết định.
        Giúp mạng hiểu 'Nếu chọn SPT thì tôi đang chọn job dài bao nhiêu'.
        """
        candidates = self.omega[~self.mask]
        rule_features = np.zeros((8, self.rule_fea_dim), dtype=np.float32)
        
        if len(candidates) == 0:
            return rule_features

        # Pre-calculate job stats to avoid repetitive computation
        # Job remaining work
        job_rem_work = np.zeros(self.number_of_jobs)
        job_ops_rem = np.zeros(self.number_of_jobs)
        
        for job_idx in range(self.number_of_jobs):
            if not self.mask[job_idx]: # If job not finished
                current_op = self.omega[job_idx]
                _, op_col = get_job_and_op_indices(current_op, self)
                # Sum duration of remaining ops including current
                job_rem_work[job_idx] = np.sum(self.dur[job_idx, op_col:])
                job_ops_rem[job_idx] = self.number_of_machines - op_col

        # Normalize factors
        max_dur = np.max(self.dur) + 1e-5
        max_rem = np.sum(self.dur) + 1e-5
        
        for r_idx in range(8):
            # 1. Giả lập rule chọn operation nào
            try:
                selected_op = apply_dispatching_rule(Rules(r_idx), candidates, self)
                job_idx, op_idx = get_job_and_op_indices(selected_op, self)
                
                # 2. Extract features của operation đó
                op_dur = self.dur[job_idx, op_idx]
                rem_work = job_rem_work[job_idx]
                ops_rem = job_ops_rem[job_idx]
                
                # 3. Fill vector (Normalized)
                rule_features[r_idx, 0] = op_dur / max_dur
                rule_features[r_idx, 1] = rem_work / max_rem
                rule_features[r_idx, 2] = ops_rem / self.number_of_machines
                
                # One-hot encoding cho rule index (để mạng biết mình đang xem xét rule nào)
                rule_features[r_idx, 3 + r_idx] = 1.0
                
            except Exception:
                # Trường hợp lỗi hoặc không có candidate (không nên xảy ra ở đây)
                pass
                
        return rule_features

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
            "mask": self.mask,
            "rule_features": self.get_rule_features() # Thêm features mới
        }

    def done(self):
        return len(self.partial_sol_sequeence) == self.number_of_tasks

    @override
    def step(self, rule_action):
        current_candidates = self.omega[~self.mask]
        
        if len(current_candidates) == 0:
            obs = self._get_obs()
            return obs, 0.0, self.done(), False, {}

        # Apply rule để lấy action thực tế
        action = apply_dispatching_rule(
            Rules(rule_action),
            current_candidates,
            self
        )
        
        # --- Logic update môi trường (giữ nguyên) ---
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

        # Reward gốc: difference in makespan
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