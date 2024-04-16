import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 第一层隐藏层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 第二层隐藏层
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)  # 第三层隐藏层
        self.fc4 = nn.Linear(hidden_dim3, output_dim)  # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
