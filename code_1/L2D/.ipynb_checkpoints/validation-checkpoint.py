import torch
import numpy as np
from pathlib import Path
from typing import Optional
from JSSP_Env import SJSSP
from mb_agg import g_pool_cal
from agent_utils import greedy_select_action
from dispatching_rules import Rules
from Params import configs

device = torch.device(configs.device)


def compute_state_features_compact(env, obs, previous_rule=None):
    """
    Computes 5 state features (4 original + 1 compact history).
    This function MUST be an exact copy of the one in PPO_jssp_multiInstances.py
    to ensure model compatibility during validation.
    """
    n_jobs = env.number_of_jobs
    n_machines = env.number_of_machines
    
    # Original 4 features
    num_active_candidates = (~obs['mask']).sum()
    fraction_complete = 1.0 - (num_active_candidates / n_jobs)
    normalized_candidates = num_active_candidates / n_jobs
    
    active_candidates = obs['candidate'][~obs['mask']]
    if len(active_candidates) > 0:
        remaining_ops = []
        for cand_id in active_candidates:
            # job_idx = cand_id // n_machines
            op_idx = cand_id % n_machines
            remaining = n_machines - op_idx
            remaining_ops.append(remaining)
        
        min_remaining = min(remaining_ops) / n_machines
        max_remaining = max(remaining_ops) / n_machines
    else:
        min_remaining = 0.0
        max_remaining = 0.0
    
    # NEW: Compact rule history - single continuous value
    if previous_rule is not None:
        # Encode as 0.0 to 1.0 (rule 0 → 0.0, rule 7 → 1.0)
        prev_rule_encoded = previous_rule / 7.0
    else:
        # No previous rule: use neutral value
        prev_rule_encoded = 0.5
    
    # Return 5 features
    return np.array([
        fraction_complete,
        normalized_candidates,
        min_remaining,
        max_remaining,
        prev_rule_encoded
    ], dtype=np.float32)


def validate(validation_data, policy, log_path: Optional[str] = None, log_probs: bool = False):
    N_JOBS = validation_data[0][0].shape[0]
    N_MACHINES = validation_data[0][0].shape[1]
    rule_names = {rule.value: rule.name for rule in Rules}
    log_handle = open(log_path, "a") if log_path else None

    env = SJSSP(n_j=N_JOBS, n_m=N_MACHINES)
    
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
        n_nodes=env.number_of_tasks,
        device=device
    )
    
    makespans = []
    
    for instance_idx, instance_data in enumerate(validation_data):
        obs, _ = env.reset(options={'data': instance_data})
        msg_header = f"[Validation] Instance {instance_idx + 1}/{len(validation_data)}"
        print(msg_header)
        if log_handle:
            log_handle.write(msg_header + "\n")
        
        done = False
        previous_rule = None  # Track previous rule for the compact feature
        
        while not done:
            fea_tensor = torch.from_numpy(np.copy(obs['fea'])).to(device)
            adj_tensor = torch.from_numpy(np.copy(obs['adj'])).to(device).to_sparse()
            
            # CHANGED: Use the compact 5-feature computation function
            state_features = compute_state_features_compact(env, obs, previous_rule)
            state_features_tensor = torch.from_numpy(state_features).float().to(device).unsqueeze(0)
            
            with torch.no_grad():
                # The policy call remains the same.
                pi, _ = policy(
                    x=fea_tensor,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    state_features=state_features_tensor,
                    return_logits=False
                )

            rule_index = greedy_select_action(pi)
            previous_rule = rule_index.item()  # Update for the next step's feature calculation
            
            step_no = env.step_count + 1
            step_msg = f"  Step {step_no}: rule {rule_names.get(rule_index.item(), 'UNKNOWN')} (index {rule_index.item()})"
            print(step_msg)
            if log_handle:
                log_handle.write(step_msg + "\n")
                if log_probs:
                    prob_msg = "    probs: " + ", ".join(f"{p:.4f}" for p in pi.squeeze().tolist())
                    log_handle.write(prob_msg + "\n")

            obs, reward, terminated, truncated, info = env.step(rule_index.item())
            done = terminated or truncated
        
        makespans.append(env.max_endTime)
    
    if log_handle:
        log_handle.flush()
        log_handle.close()
    return np.array(makespans)