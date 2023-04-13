import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from model.Network import Critic, Transformer


class Memory:
    def __init__(self):
        self.input_x = []
        self.input_attr = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def def_memory(self):
        self.input_x.clear()
        self.input_attr.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()


class Actor_Critic(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers):
        super(Actor_Critic, self).__init__()
        self.actor = Transformer(input_node_dim=input_node_dim,
                                 hidden_node_dim=hidden_node_dim,
                                 input_edge_dim=input_edge_dim,
                                 hidden_edge_dim=hidden_edge_dim,
                                 conv_layers=conv_layers)

        self.critic = Critic(hidden_node_dim)

    def act(self, datas, actions, steps, greedy, _action):
        actions, log_p, _, _, _ = self.actor(datas, actions, steps, greedy, _action)
        return actions, log_p

    def evaluate(self, datas, actions, steps, greedy, _action):
        _, _, entropy, old_log_p, x = self.actor(datas, actions, steps, greedy, _action)
        value = self.critic(x)
        return entropy, old_log_p, value


class PPO:
    def __init__(self,
                 node_number: int = 20,
                 lr: float = 0.0003,
                 input_node_dim: int = 2,
                 hidden_node_dim: int = 128,
                 input_edge_dim: int = 1,
                 hidden_edge_dim: int = 16,
                 ppo_epoch: int = 1,
                 batch_size: int = 32,
                 conv_layers: int = 3,
                 entropy_value: float = 0.2,
                 eps_clip: float = 0.2,
                 device: str = 'cpu'):

        self.policy = Actor_Critic(input_node_dim=input_node_dim,
                                   hidden_node_dim=hidden_node_dim,
                                   input_edge_dim=input_edge_dim,
                                   hidden_edge_dim=hidden_edge_dim,
                                   conv_layers=conv_layers)

        self.main_policy = Actor_Critic(input_node_dim=input_node_dim,
                                        hidden_node_dim=hidden_node_dim,
                                        input_edge_dim=input_edge_dim,
                                        hidden_edge_dim=hidden_edge_dim,
                                        conv_layers=conv_layers)

        self.main_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.batch_size = batch_size
        self.epoch = ppo_epoch
        self.node_number = node_number
        self.entropy_value = entropy_value
        self.eps_clip = eps_clip
        self.device = device
        self.edges_index = self.get_edges_index()
        self.lr = lr

    def load_parameters(self):
        self.main_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def get_edges_index(self):
        edges_index = []
        for i in range(self.node_number):
            for j in range(self.node_number):
                edges_index.append([i, j])
        edges_index = torch.LongTensor(edges_index)
        edges_index = edges_index.transpose(dim0=0, dim1=1)
        return edges_index

    def adv_normalize(self, adv):
        std = adv.std()
        assert std != 0. and not torch.isnan(std), 'Need nonzero std'
        n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
        return n_advs

    def value_loss_gae(self, val_targ, old_vs, value_od, clip_val):
        vs_clipped = old_vs + torch.clamp(old_vs - value_od, -clip_val, +clip_val)
        val_loss_mat_unclipped = self.MseLoss(old_vs, val_targ)
        val_loss_mat_clipped = self.MseLoss(vs_clipped, val_targ)
        val_loss_mat = torch.max(val_loss_mat_unclipped, val_loss_mat_clipped)
        mse = val_loss_mat
        return mse

    def update(self, memory, epoch):
        old_input_x = torch.stack(memory.input_x)
        old_input_attr = torch.stack(memory.input_attr)
        old_action = torch.stack(memory.actions)
        old_rewards = torch.stack(memory.rewards).unsqueeze(-1)
        old_log_probs = torch.stack(memory.log_probs).unsqueeze(-1)

        datas = []
        for i in range(old_input_x.size(0)):
            data = Data(
                x=old_input_x[i],
                edge_index=self.edges_index,
                edge_attr=old_input_attr[i],
                actions=old_action[i],
                rewards=old_rewards[i],
                log_probs=old_log_probs[i]
            )
            datas.append(data)

        self.policy.to(self.device)
        data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=False)
        # 学习率退火
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda f: 0.96 ** epoch)

        for i in range(self.epoch):

            self.policy.train()

            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(self.device)

                entropy, log_probs, value = self.policy.evaluate(batch, batch.actions, self.node_number, False, True)

                # advangtage function
                rewar = self.adv_normalize(batch.rewards)

                # Value function clipping
                mse_loss = self.MseLoss(rewar, value)

                ratios = torch.exp(log_probs - batch.log_probs)

                # norm advantages
                advantages = rewar - value.detach()

                # PPO loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # total loss
                loss = torch.min(surr1, surr2) + 0.5 * mse_loss - self.entropy_value * entropy
                self.optimizer.zero_grad()
                loss.mean().backward()

                # max_grad_norm=2
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 2)
                self.optimizer.step()

                scheduler.step()

        self.main_policy.load_state_dict(self.policy.state_dict())

# evaluate(self.greedy, True)
# rollout(True, False)
