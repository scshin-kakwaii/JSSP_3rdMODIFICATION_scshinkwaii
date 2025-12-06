"""
actor_critic.py - Modified for 5-feature state (was 12 features)

Key changes:
1. Actor input: hidden_dim + 5 (was hidden_dim + 12)
2. Can return raw logits for temperature scaling
3. Handles both training and validation modes
"""
import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
import torch


class ActorCritic(nn.Module):
    """
    5-Feature Version: Accepts compact state features
    
    State features (5 total):
    - fraction_complete (0-1)
    - normalized_candidates (0-1)
    - min_remaining (0-1)
    - max_remaining (0-1)
    - previous_rule_encoded (0-1)  ← Was 8 one-hot features
    """
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # critic net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(ActorCritic, self).__init__()
        # job size for problems
        self.n_j = n_j
        # machine size for problems
        self.n_m = n_m
        self.device = device

        # There are 8 dispatching rules
        self.num_rules = 8

        # GNN for graph feature extraction
        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        
        # CHANGED: 5 state features (was 12 or 4)
        num_state_features = 5
        
        # Actor: takes global graph features + state features
        actor_input_dim = hidden_dim + num_state_features
        
        self.actor = MLPActor(num_mlp_layers_actor, 
                              actor_input_dim,
                              hidden_dim_actor, 
                              self.num_rules).to(device)

        # Critic: only takes global graph features (no state features needed)
        self.critic = MLPCritic(num_mlp_layers_critic, 
                               hidden_dim, 
                               hidden_dim_critic, 
                               1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate=None,      # Not used in rule-based version
                mask=None,           # Not used in rule-based version
                state_features=None, # NEW: 5-feature state
                return_logits=False  # NEW: For temperature scaling
                ):
        """
        Forward pass
        
        Args:
            x: Node features [batch*n_ops, input_dim]
            graph_pool: Pooling matrix for graph-level features
            padded_nei: Padded neighbor indices (not used)
            adj: Adjacency matrix [batch*n_ops, batch*n_ops]
            candidate: Not used (kept for API compatibility)
            mask: Not used (kept for API compatibility)
            state_features: [batch, 5] state features
            return_logits: If True, return raw logits before softmax
        
        Returns:
            pi: Policy distribution [batch, 8] or logits if return_logits=True
            v: Value estimate [batch, 1]
        """
        # 1. Extract graph features
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        # h_pooled: [batch, hidden_dim] - global graph representation
        # h_nodes: [batch*n_ops, hidden_dim] - per-node features (not used for rules)
        
        batch_size = h_pooled.shape[0]
        
        # 2. Prepare actor input (global features + state features)
        if state_features is not None:
            # Concatenate: [batch, hidden_dim] + [batch, 5] → [batch, hidden_dim+5]
            actor_input = torch.cat([h_pooled, state_features], dim=-1)
        else:
            # Fallback: use zero state features
            zero_features = torch.zeros(batch_size, 5, device=self.device)
            actor_input = torch.cat([h_pooled, zero_features], dim=-1)
        
        # 3. Get rule scores from actor
        rule_scores = self.actor(actor_input)  # [batch, 8]
        
        # 4. Return logits or probabilities
        if return_logits:
            # For temperature scaling during training
            v = self.critic(h_pooled)
            return rule_scores, v
        else:
            # Normal forward pass: return probabilities
            pi = F.softmax(rule_scores, dim=-1)  # [batch, 8]
            v = self.critic(h_pooled)  # [batch, 1]
            return pi, v


if __name__ == '__main__':
    print('ActorCritic for 5-feature rule selection')
    print('Changes from original:')
    print('  - State features: 5 (was 12 with one-hot, or 4 without history)')
    print('  - Actor input: hidden_dim + 5')
    print('  - Can return raw logits for temperature scaling')