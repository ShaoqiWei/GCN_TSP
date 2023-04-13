import torch.nn as nn
import torch.nn.functional as F
from GCN_TSP.network.Layers import Encoder, Decoder


class Transformer(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_node_dim=input_node_dim,
                               hidden_node_dim=hidden_node_dim,
                               input_edge_dim=input_edge_dim,
                               hidden_edge_dim=hidden_edge_dim,
                               conv_layers=conv_layers)

        self.decoder = Decoder(input_dim=hidden_node_dim,
                               hidden_dim=hidden_node_dim)

    def forward(self, datas, actions_old, n_steps, greedy, _action):
        # (batch,seq_len,hidden_node_dim)
        x = self.encoder(datas)
        pooled = x.mean(dim=1)
        actions, log_p, entropy, dists = self.decoder(x,
                                                      pooled,
                                                      actions_old=actions_old,
                                                      n_steps=n_steps,
                                                      greedy=greedy,
                                                      _action=_action)
        return actions, log_p, entropy, dists, x


class Critic(nn.Module):
    def __init__(self, hidden_node_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Conv1d(hidden_node_dim, 20, kernel_size=(1,))
        self.fc2 = nn.Conv1d(20, 20, kernel_size=(1,))
        self.fc3 = nn.Conv1d(20, 1, kernel_size=(1,))

    def forward(self, x):
        x1 = x.transpose(2, 1)
        output = F.relu(self.fc1(x1))
        output = F.relu(self.fc2(output))
        value = self.fc3(output).sum(dim=2).squeeze(-1)
        return value
