"""
COMPLETE 5-FEATURE VERSION
Changes from 12-feature version:
- Rule history: 8 one-hot → 1 continuous value (0-1)
- Total features: 12 → 5
- More balanced, less self-reinforcing
"""
from mb_agg import *
from agent_utils import eval_actions, select_action
from models.actor_critic import ActorCritic
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from Params import configs
from validation import validate

device = torch.device(configs.device)


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.state_features_mb = []  # Now 5 features (was 12)
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.state_features_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


def compute_state_features_compact(env, obs, previous_rule=None):
    """
    Compute 5 state features (CHANGED from 12):
    - 4 original features (progress, candidates, min/max remaining)
    - 1 compact rule history (single value 0-1, not one-hot)
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
            job_idx = cand_id // n_machines
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
    
    # Return 5 features (was 12)
    return np.array([
        fraction_complete,
        normalized_candidates,
        min_remaining,
        max_remaining,
        prev_rule_encoded
    ], dtype=np.float32)


class ActorCriticCompact(nn.Module):
    """
    CHANGED: Now accepts 5 state features instead of 12
    """
    def __init__(self, n_j, n_m, num_layers, learn_eps, neighbor_pooling_type,
                 input_dim, hidden_dim, num_mlp_layers_feature_extract,
                 num_mlp_layers_actor, hidden_dim_actor,
                 num_mlp_layers_critic, hidden_dim_critic, device):
        super(ActorCriticCompact, self).__init__()
        from models.graphcnn_congForSJSSP import GraphCNN
        from models.mlp import MLPActor, MLPCritic
        
        self.n_j = n_j
        self.n_m = n_m
        self.device = device
        self.num_rules = 8

        self.feature_extract = GraphCNN(
            num_layers=num_layers,
            num_mlp_layers=num_mlp_layers_feature_extract,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learn_eps=learn_eps,
            neighbor_pooling_type=neighbor_pooling_type,
            device=device
        ).to(device)
        
        # CHANGED: 5 state features (was 12)
        num_state_features = 5
        actor_input_dim = hidden_dim + num_state_features
        
        self.actor = MLPActor(num_mlp_layers_actor, actor_input_dim,
                              hidden_dim_actor, self.num_rules).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim,
                               hidden_dim_critic, 1).to(device)

    def forward(self, x, graph_pool, padded_nei, adj, state_features=None, return_logits=False):
        h_pooled, h_nodes = self.feature_extract(
            x=x, graph_pool=graph_pool, padded_nei=padded_nei, adj=adj
        )
        
        batch_size = h_pooled.shape[0]
        
        if state_features is not None:
            actor_input = torch.cat([h_pooled, state_features], dim=-1)
        else:
            # CHANGED: 5 features (was 12)
            zero_features = torch.zeros(batch_size, 5, device=self.device)
            actor_input = torch.cat([h_pooled, zero_features], dim=-1)
        
        rule_scores = self.actor(actor_input)
        
        if return_logits:
            return rule_scores, self.critic(h_pooled)
        
        pi = torch.nn.functional.softmax(rule_scores, dim=-1)
        v = self.critic(h_pooled)
        
        return pi, v


class PPO:
    def __init__(self, lr, gamma, k_epochs, eps_clip, n_j, n_m,
                 num_layers, neighbor_pooling_type, input_dim, hidden_dim,
                 num_mlp_layers_feature_extract, num_mlp_layers_actor,
                 hidden_dim_actor, num_mlp_layers_critic, hidden_dim_critic):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # CHANGED: Use ActorCriticCompact (5 features)
        self.policy = ActorCriticCompact(
            n_j=n_j, n_m=n_m, num_layers=num_layers, learn_eps=False,
            neighbor_pooling_type=neighbor_pooling_type,
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
            num_mlp_layers_actor=num_mlp_layers_actor,
            hidden_dim_actor=hidden_dim_actor,
            num_mlp_layers_critic=num_mlp_layers_critic,
            hidden_dim_critic=hidden_dim_critic,
            device=device
        )
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio
        )
        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        state_features_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), 
                                           reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            
            adj_mb_t_all_env.append(
                aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks)
            )
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            
            state_features_mb_t_all_env.append(
                torch.stack(memories[i].state_features_mb).to(device)
            )
            
            a_mb_t_all_env.append(
                torch.stack(memories[i].a_mb).to(device).squeeze()
            )
            old_logprobs_mb_t_all_env.append(
                torch.stack(memories[i].logprobs).to(device).squeeze().detach()
            )

        mb_g_pool = g_pool_cal(
            g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device
        )

        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(
                    x=fea_mb_t_all_env[i],
                    graph_pool=mb_g_pool,
                    adj=adj_mb_t_all_env[i],
                    state_features=state_features_mb_t_all_env[i],
                    padded_nei=None,
                    return_logits=False
                )
                
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.view(-1).detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 
                                   1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2).mean()
                ent_loss = - ent_loss.clone()
                
                # Diversity penalty
                max_probs = pis.max(dim=-1)[0]
                diversity_penalty = torch.relu(max_probs - 0.7).mean() * 2.0
                
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss + diversity_penalty
                loss_sum += loss
                vloss_sum += v_loss
                
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()


def get_adaptive_temperature(episode, current_quality, best_recent_quality, quality_history):
    """Adaptive temperature based on performance"""
    # Base temperature schedule
    if episode < 100:
        base_temp = 10.0
    elif episode < 300:
        base_temp = 10.0 - (episode - 100) / 200 * 7.0
    elif episode < 600:
        base_temp = 3.0 - (episode - 300) / 300 * 1.5
    elif episode < 1000:
        base_temp = 1.5 - (episode - 600) / 400 * 0.5
    else:
        base_temp = 1.0
    
    # Don't adapt for first 300 episodes
    if episode < 300 or len(quality_history) < 5:
        return base_temp
    
    recent_avg = sum(quality_history[-5:]) / 5
    
    # LOCK IN: Current quality is great
    if current_quality <= best_recent_quality * 1.02:
        reduced_temp = max(base_temp * 0.5, 1.0)
        return reduced_temp
    
    # ESCAPE: Performance degrading
    elif current_quality > recent_avg * 1.10:
        increased_temp = min(base_temp * 1.3, 3.0)
        return increased_temp
    
    return base_temp


def main():
    from JSSP_Env import SJSSP
    from uniform_instance_gen import uni_instance_gen
    
    # High entropy to prevent collapse
    print(f"Original entropy coef: {configs.entloss_coef}")
    configs.entloss_coef = 0.20  # Slightly lower than 0.25 since features are more balanced
    print(f"New entropy coef: {configs.entloss_coef}")
    
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m) for _ in range(configs.num_envs)]
    data_generator = uni_instance_gen

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / f'DataGen/generatedData{configs.n_j}_{configs.n_m}_Seed{configs.np_seed_validation}.npy'
    dataLoaded = np.load(data_path)
    vali_data = [(dataLoaded[i][0], dataLoaded[i][1]) for i in range(dataLoaded.shape[0])]

    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j, n_m=configs.n_m,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim, hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    rule_log_path = Path(__file__).resolve().parent / "rule_trace_5features.txt"
    rule_log_path.write_text("Rule trace with 5 compact features\n")

    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, configs.n_j*configs.n_m, configs.n_j*configs.n_m]),
        n_nodes=configs.n_j*configs.n_m,
        device=device
    )
    
    log = []
    validation_log = []
    quality_history = []
    record = 100000
    best_recent = 100000
    # REMOVED: Early stopping
    # patience_counter = 0
    # patience_limit = 15
    
    # Initialize
    states = []
    previous_rules = []
    for i, env in enumerate(envs):
        instance_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, 
                                      low=configs.low, high=configs.high)
        obs, _ = env.reset(options={'data': instance_data})
        states.append(obs)
        previous_rules.append(None)

    for i_update in range(configs.max_updates):
        ep_rewards = [-env.initQuality for env in envs]
        
        # Adaptive temperature
        temperature = get_adaptive_temperature(
            i_update,
            validation_log[-1] if validation_log else 1000,
            best_recent,
            quality_history
        )

        # Rollout
        while True:
            fea_tensors = [torch.from_numpy(np.copy(s['fea'])).to(device) for s in states]
            adj_tensors = [torch.from_numpy(np.copy(s['adj'])).to(device).to_sparse() 
                          for s in states]
            
            # CHANGED: Use compact state features (5 instead of 12)
            state_features = [
                compute_state_features_compact(envs[i], states[i], previous_rules[i])
                for i in range(configs.num_envs)
            ]
            state_feature_tensors = [torch.from_numpy(sf).float().to(device) for sf in state_features]
            
            with torch.no_grad():
                rule_indices = []
                for i in range(configs.num_envs):
                    logits, _ = ppo.policy_old(
                        x=fea_tensors[i],
                        graph_pool=g_pool_step,
                        padded_nei=None,
                        adj=adj_tensors[i],
                        state_features=state_feature_tensors[i].unsqueeze(0),
                        return_logits=True
                    )
                    
                    tempered_logits = logits / temperature
                    tempered_pi = torch.softmax(tempered_logits, dim=-1)
                    
                    rule_idx, _ = select_action(tempered_pi, memories[i])
                    rule_indices.append(rule_idx)
            
            next_states = []
            next_previous_rules = []
            
            for i in range(configs.num_envs):
                memories[i].adj_mb.append(adj_tensors[i])
                memories[i].fea_mb.append(fea_tensors[i])
                memories[i].state_features_mb.append(state_feature_tensors[i])
                memories[i].a_mb.append(rule_indices[i])

                obs, reward, terminated, truncated, _ = envs[i].step(rule_indices[i].item())
                done = terminated or truncated
                next_states.append(obs)
                next_previous_rules.append(rule_indices[i].item())

                ep_rewards[i] += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)

            states = next_states
            previous_rules = next_previous_rules

            if done:
                states = []
                previous_rules = []
                for i, env in enumerate(envs):
                    instance_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, 
                                                  low=configs.low, high=configs.high)
                    obs, _ = env.reset(options={'data': instance_data})
                    states.append(obs)
                    previous_rules.append(None)
                break
                
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs[j].posRewards

        loss, v_loss = ppo.update(memories, configs.n_j*configs.n_m, configs.graph_pool_type)
        for memory in memories:
            memory.clear_memory()
            
        mean_reward = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_reward])
        
        if (i_update + 1) % 10 == 0:
            print(f'Ep {i_update+1:4d} | Reward: {mean_reward:7.2f} | VLoss: {v_loss:.6f} | Temp: {temperature:.2f}')
            
            # Policy diversity check
            if (i_update + 1) % 50 == 0:
                with torch.no_grad():
                    sample_fea = torch.from_numpy(np.copy(states[0]['fea'])).to(device)
                    sample_adj = torch.from_numpy(np.copy(states[0]['adj'])).to(device).to_sparse()
                    sample_sf = compute_state_features_compact(envs[0], states[0], previous_rules[0])
                    sample_sf_tensor = torch.from_numpy(sample_sf).float().to(device).unsqueeze(0)
                    
                    sample_pi, _ = ppo.policy(
                        x=sample_fea, graph_pool=g_pool_step, padded_nei=None,
                        adj=sample_adj, state_features=sample_sf_tensor, return_logits=False
                    )
                    probs = sample_pi.squeeze().cpu().numpy()
                    max_prob = probs.max()
                    entropy = -(probs * np.log(probs + 1e-10)).sum()
                    
                    print(f'  Policy: MaxProb={max_prob:.3f}, Entropy={entropy:.3f}/2.08')
                    if max_prob > 0.8:
                        print(f'  ⚠️  Collapsing! Probs: {" ".join(f"{p:.2f}" for p in probs)}')
        
        if (i_update + 1) % 100 == 0:
            with open(rule_log_path, "a") as f:
                f.write(f"\n=== Episode {i_update + 1} ===\n")
            vali_result = validate(vali_data, ppo.policy, log_path=str(rule_log_path), 
                                  log_probs=True).mean()
            validation_log.append(vali_result)
            quality_history.append(vali_result)
            
            # Update best recent
            if len(quality_history) > 10:
                best_recent = min(quality_history[-10:])
            else:
                best_recent = min(quality_history)
            
            # Save if best overall
            if vali_result < record:
                record = vali_result
                # REMOVED: patience_counter = 0
                torch.save(ppo.policy.state_dict(), 
                          f'./{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}_5feat_BEST.pth')
                print(f'>>> VALIDATION: {vali_result:.2f} (NEW BEST! ⭐)')
            else:
                # REMOVED: patience_counter += 1
                print(f'>>> VALIDATION: {vali_result:.2f} (best: {record:.2f})')
            
            # Periodic checkpoints
            if (i_update + 1) % 500 == 0:
                torch.save(ppo.policy.state_dict(), 
                          f'./{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}_5feat_ep{i_update+1}.pth')
                print(f'  💾 Checkpoint saved')
            
            # REMOVED: Early stopping
            # if patience_counter >= patience_limit:
            #     print(f"\n🛑 Early stopping at episode {i_update + 1}")
            #     print(f"Best validation: {record:.2f}")
            #     break


if __name__ == '__main__':
    main()