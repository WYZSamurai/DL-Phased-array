import torch
import MLP
import data
import lossfunc


def train(model: MLP.MLP, device: torch.device,  optimizer: torch.optim.Adam, num_epochs: int, batch_size: int, delta: int, lamb: float, d: float, theta_0: float, best_model_path: str):
    # 最佳损失值
    best_loss = float("inf")
    # 训练模式
    model.train()

    for epoch in range(num_epochs):
        # 生成数据
        inputs, mask = data.generate(batch_size, delta)
        inputs = inputs.to(device)
        # 优化器清零梯度
        optimizer.zero_grad()
        # 输出参数
        outputs: torch.Tensor = model(inputs)
        # 阵元数
        NE = int(outputs.shape[1]/2)
        # 生成Fdb数据
        transformed_outputs = data.pattern(
            outputs[:, :NE], outputs[:, NE:], lamb, d, delta, theta_0)
        # 计算损失值
        loss = lossfunc.total_loss(transformed_outputs, inputs, mask)
        # loss: torch.Tensor = criterion(transformed_outputs, inputs)
        # 向后传播
        loss.backward()
        # 优化器步进
        optimizer.step()
        # 保存最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), best_model_path)
            print("最佳模型更新")

        print(f"已迭代 {epoch+1}代, 损失值: {loss.item()}")


def evaluate(model: MLP.MLP, batch_size: int, delta: int, model_path: str, lamb: float, d: float, theta_0: float, device: torch.device):
    # 载入模型
    model.load_state_dict(torch.load(model_path))
    # 评估模式
    model.eval()
    with torch.no_grad():
        # 生成测试数据
        inputs, mask = data.generate(batch_size, delta)
        inputs = inputs.to(device)
        # 生成结果
        outputs: torch.Tensor = model(inputs)
        NE = int(outputs.shape[1]/2)
        # 转换结果
        transformed_outputs = data.pattern(
            outputs[:, :NE], outputs[:, NE:], lamb, d, delta, theta_0)
        # 计算损失值
        loss = lossfunc.total_loss(transformed_outputs, inputs, mask)
        # loss: torch.Tensor = criterion(transformed_outputs, inputs)
    # 绘制第一个方向图的结果
    data.plot(inputs[0])
    data.plot(transformed_outputs[0])
    print(f'评估损失值: {loss.item()}')
