import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_heads=4):
        super(GATNet, self).__init__()

        self.attention1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.attention2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)

        self.fc = nn.Linear(hidden_channels * num_heads, out_channels)

        self.residual_fc = nn.Linear(in_channels, hidden_channels * num_heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        residual = self.residual_fc(x)
        x = F.relu(self.attention1(x, edge_index))
        x = F.relu(self.attention2(x, edge_index))
        x = x + residual
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)

        return F.log_softmax(x, dim=1), 1


if __name__ == '__main__':
    model = GATNet(in_channels=38, out_channels=2)
    print(model)
