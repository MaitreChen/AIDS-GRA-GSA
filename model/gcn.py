import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(GCNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)

        return F.log_softmax(x, dim=1), 1


if __name__ == '__main__':
    model = GCNet(in_channels=38, out_channels=2)
    print(model)
