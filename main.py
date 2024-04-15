# x(batch_size,delta)
# y(batch_size,2*NE)


import torch
import MLP
import train


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")


# theta范围
# theta_max = 90.0
# theta_min = -90.0


theta0 = 0.0
lamb = 1.0
d = 0.5*lamb
# 缩放倍数
scale = 1
delta = scale*180
# 训练批数
batch_size = 1000
# 阵元数
NE = 24


# 模型
model = MLP.MLP(delta, 128, 64, 32, 2*NE).to(device)
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


best_model_path = "best model.pth"


# 训练过程
num_epochs = 1000
train.train(model, device, optimizer, num_epochs,
            batch_size, delta, lamb, d, theta0, best_model_path)


# 评估过程
train.evaluate(model, batch_size, delta,
               best_model_path, lamb, d, theta0, device)
