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
        self.candidate_mb = []
        self.mask_mb = []
        self.rule_fea_mb = [] # New: Store rule features
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.rule_fea_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]

class PPO:
    def __init__(self, lr, gamma, k_epochs, eps_clip, n_j, n_m,
                 num_layers, neighbor_pooling_type, input_dim, hidden_dim,
                 num_mlp_layers_feature_extract, num_mlp_layers_actor,
                 hidden_dim_actor, num_mlp_layers_critic, hidden_dim_critic):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(n_j=n_j, n_m=n_m, num_layers=num_layers, learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim, hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio
        )
        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool, current_ent_coef):
        self.policy.train() 
        
        all_rewards = []
        all_adj = []
        all_fea = []
        all_cand = []
        all_mask = []
        all_rule_fea = []
        all_actions = []
        all_old_probs = []
        
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            
            all_rewards.extend(rewards)
            all_adj.extend(memories[i].adj_mb)
            all_fea.extend(memories[i].fea_mb)
            all_cand.extend(memories[i].candidate_mb)
            all_mask.extend(memories[i].mask_mb)
            all_rule_fea.extend(memories[i].rule_fea_mb)
            all_actions.extend(memories[i].a_mb)
            all_old_probs.extend(memories[i].logprobs)

        batch_rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-5)
        
        batch_fea = torch.stack(all_fea).to(device)
        batch_cand = torch.stack(all_cand).to(device)
        batch_mask = torch.stack(all_mask).to(device)
        batch_rule_fea = torch.stack(all_rule_fea).to(device)
        batch_actions = torch.stack(all_actions).to(device).detach()
        batch_old_probs = torch.stack(all_old_probs).to(device).detach()
        
        total_steps = len(all_adj)
        indices = np.arange(total_steps)
        
        mini_batch_size = 64
        avg_loss = 0
        
        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, total_steps, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]
                
                mb_fea = batch_fea[mb_idx].reshape(-1, batch_fea.size(-1)) 

                mb_cand = batch_cand[mb_idx]
                mb_mask = batch_mask[mb_idx]
                mb_rule_fea = batch_rule_fea[mb_idx]
                mb_actions = batch_actions[mb_idx]
                mb_old_probs = batch_old_probs[mb_idx]
                mb_rewards = batch_rewards[mb_idx]
                
                mb_adj_list = [all_adj[i] for i in mb_idx]
                mb_adj_tensor = torch.stack(mb_adj_list).to(device)
                
                mb_adj_batch = aggr_obs(mb_adj_tensor.to_sparse(), n_tasks)
                
                mb_g_pool = g_pool_cal(configs.graph_pool_type, 
                                     torch.Size([len(mb_idx), n_tasks*n_tasks, n_tasks*n_tasks]), 
                                     n_tasks, device)
                
                pis, vals = self.policy(x=mb_fea,
                                        graph_pool=mb_g_pool,
                                        padded_nei=None,
                                        adj=mb_adj_batch,
                                        candidates=mb_cand,
                                        mask=mb_mask,
                                        rule_features=mb_rule_fea)
                
                logprobs, ent_loss = eval_actions(pis, mb_actions)
                ratios = torch.exp(logprobs - mb_old_probs)
                advantages = mb_rewards - vals.view(-1).detach()
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                
                v_loss = self.V_loss_2(vals.squeeze(), mb_rewards)
                p_loss = -torch.min(surr1, surr2).mean()
                
                loss = configs.vloss_coef * v_loss + configs.ploss_coef * p_loss - current_ent_coef * ent_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                avg_loss += loss.item()
                
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
            
        return avg_loss / (self.k_epochs * (total_steps // mini_batch_size + 1)), 0

def main():
    from JSSP_Env import SJSSP
    from uniform_instance_gen import uni_instance_gen

    print(f"Original entropy coef: {configs.entloss_coef}")
    configs.entloss_coef = 0.20
    print(f"New entropy coef: {configs.entloss_coef}")
    
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m) for _ in range(configs.num_envs)]
    data_generator = uni_instance_gen
    
    # Tạo folder nếu chưa có
    Path('./DataGen').mkdir(parents=True, exist_ok=True)
    
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / f'DataGen/generatedData{configs.n_j}_{configs.n_m}_Seed{configs.np_seed_validation}.npy'
    
    # Handle loading data gracefully
    if data_path.exists():
        dataLoaded = np.load(data_path)
        vali_data = [(dataLoaded[i][0], dataLoaded[i][1]) for i in range(dataLoaded.shape[0])]
    else:
        print(f"Warning: Validation data not found at {data_path}. Generating new validation data.")
        # Generate temporary validation data
        vali_data = [data_generator(configs.n_j, configs.n_m, configs.low, configs.high) for _ in range(10)]

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
    rule_log_path.write_text("Rule trace with refactored logic\n")

    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, configs.n_j*configs.n_m, configs.n_j*configs.n_m]),
        n_nodes=configs.n_j*configs.n_m,
        device=device
    )
    
    # --- KHỞI TẠO CÁC BIẾN LOGGING (ĐÃ FIX LỖI NAME ERROR) ---
    log = []
    validation_log = []
    quality_history = []  # <--- THÊM DÒNG NÀY ĐỂ FIX LỖI TẠI EP 100
    record = 100000
    # ---------------------------------------------------------
    
    states = []
    for i, env in enumerate(envs):
        instance_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, 
                                      low=configs.low, high=configs.high)
        obs, _ = env.reset(options={'data': instance_data})
        states.append(obs)

    for i_update in range(configs.max_updates):
        frac = 1.0 - (i_update - 1.0) / configs.max_updates
        current_ent_coef = max(configs.entloss_coef * frac, 0.005)
        
        ppo.policy_old.eval()
        
        ep_rewards = [0 for _ in envs]
        
        while True:
            fea_tensors = [torch.from_numpy(s['fea']).to(device) for s in states]
            adj_tensors = [torch.from_numpy(s['adj']).to(device) for s in states]
            cand_tensors = [torch.from_numpy(s['candidate']).to(device) for s in states]
            mask_tensors = [torch.from_numpy(s['mask']).to(device) for s in states]
            rule_fea_tensors = [torch.from_numpy(s['rule_features']).to(device) for s in states]
            
            with torch.no_grad():
                rule_indices = []
                for i in range(configs.num_envs):
                    pi, _ = ppo.policy_old(x=fea_tensors[i],
                                           graph_pool=g_pool_step,
                                           padded_nei=None,
                                           adj=adj_tensors[i].to_sparse(),
                                           candidates=cand_tensors[i].unsqueeze(0),
                                           mask=mask_tensors[i].unsqueeze(0),
                                           rule_features=rule_fea_tensors[i].unsqueeze(0))
                    
                    action, _ = select_action(pi, memories[i])
                    rule_indices.append(action)

            next_states = []
            done_any = False
            
            for i in range(configs.num_envs):
                memories[i].adj_mb.append(adj_tensors[i])
                memories[i].fea_mb.append(fea_tensors[i])
                memories[i].candidate_mb.append(cand_tensors[i])
                memories[i].mask_mb.append(mask_tensors[i])
                memories[i].rule_fea_mb.append(rule_fea_tensors[i])
                memories[i].a_mb.append(rule_indices[i])

                obs, reward, terminated, truncated, _ = envs[i].step(rule_indices[i].item())
                done = terminated or truncated
                next_states.append(obs)
                
                ep_rewards[i] += reward
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
                
                if done: done_any = True

            states = next_states

            if done_any:
                states = []
                for i, env in enumerate(envs):
                    instance_data = data_generator(n_j=configs.n_j, n_m=configs.n_m, 
                                                  low=configs.low, high=configs.high)
                    obs, _ = env.reset(options={'data': instance_data})
                    states.append(obs)
                break
        
        loss, _ = ppo.update(memories, configs.n_j*configs.n_m, configs.graph_pool_type, current_ent_coef)
        
        for memory in memories:
            memory.clear_memory()
            
        mean_reward = sum(ep_rewards) / len(ep_rewards)
        log.append([i_update, mean_reward])
        
        if (i_update + 1) % 10 == 0:
            print(f'Ep {i_update+1:4d} | Reward: {mean_reward:7.2f} | Loss: {loss:.4f} | EntCoef: {current_ent_coef:.4f}')
        
        if (i_update + 1) % 100 == 0:
            with open(rule_log_path, "a") as f:
                f.write(f"\n=== Episode {i_update + 1} ===\n")
            
            # Hàm validate này yêu cầu file validation.py phải được cập nhật
            vali_result = validate(vali_data, ppo.policy, log_path=str(rule_log_path), log_probs=True).mean()
            
            validation_log.append(vali_result)
            quality_history.append(vali_result) # Đã fix
            
            if len(quality_history) > 10:
                best_recent = min(quality_history[-10:])
            else:
                best_recent = min(quality_history)
            
            if vali_result < record:
                record = vali_result
                torch.save(ppo.policy.state_dict(), 
                          f'./{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}_BestLogic.pth')
                print(f'>>> VALIDATION: {vali_result:.2f} (NEW BEST! ⭐)')
            else:
                print(f'>>> VALIDATION: {vali_result:.2f} (best: {record:.2f})')
            
            if (i_update + 1) % 500 == 0:
                torch.save(ppo.policy.state_dict(), 
                          f'./{configs.n_j}_{configs.n_m}_{configs.low}_{configs.high}_ep{i_update+1}.pth')

if __name__ == '__main__':
    main()