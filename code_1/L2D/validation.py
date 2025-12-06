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

def validate(validation_data, policy, log_path: Optional[str] = None, log_probs: bool = False):
    N_JOBS = validation_data[0][0].shape[0]
    N_MACHINES = validation_data[0][0].shape[1]
    rule_names = {rule.value: rule.name for rule in Rules}
    log_handle = open(log_path, "a") if log_path else None
    
    # Switch policy to eval mode for validation
    policy.eval()

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
        
        while not done:
            # Prepare inputs - Updated for Action-Aware Architecture
            fea_tensor = torch.from_numpy(np.copy(obs['fea'])).to(device)
            adj_tensor = torch.from_numpy(np.copy(obs['adj'])).to(device).to_sparse()
            cand_tensor = torch.from_numpy(np.copy(obs['candidate'])).to(device).unsqueeze(0)
            mask_tensor = torch.from_numpy(np.copy(obs['mask'])).to(device).unsqueeze(0)
            rule_fea_tensor = torch.from_numpy(np.copy(obs['rule_features'])).to(device).unsqueeze(0)
            
            with torch.no_grad():
                pi, _ = policy(
                    x=fea_tensor,
                    graph_pool=g_pool_step,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidates=cand_tensor,
                    mask=mask_tensor,
                    rule_features=rule_fea_tensor
                )

            rule_index = greedy_select_action(pi)
            
            step_no = env.step_count + 1
            step_msg = f"  Step {step_no}: rule {rule_names.get(rule_index.item(), 'UNKNOWN')} (index {rule_index.item()})"
            # print(step_msg) # Uncomment to spam console
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