import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GRANet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_heads=4):
        super(GRANet, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.attention = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=True)

        self.fc = nn.Linear(hidden_channels * num_heads, out_channels)

        self.residual_fc = nn.Linear(in_channels, hidden_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        residual = self.residual_fc(x)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = x + residual

        x, attention_weights = self.attention(x, edge_index, return_attention_weights=True)
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)

        x = self.fc(x)

        return F.log_softmax(x, dim=1),attention_weights


if __name__ == '__main__':
    model = GRANet(in_channels=38, out_channels=2)
    print(model)
