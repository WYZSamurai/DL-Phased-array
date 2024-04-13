from MLP import MLP
import data
import torch
import plotly.graph_objects as go


def train():
    # 建立深度网络模型
    model = MLP().double()
    # 数据生成
    x_train, x_test, y_train, y_test = data.spl_data(4)
    # 优化器选择
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2.5e-5, weight_decay=1e-5)
    # 损失函数选择
    lossfunction = torch.nn.MSELoss()

    # 设置迭代周期数
    epochs = 1000
    sum_loss = 0
    temp = 1000
    # 记录每次迭代的loss值
    loss_y = torch.zeros(size=(epochs,), dtype=torch.float64)
    for epoch in range(epochs):
        # 前向传播
        y_pre = model(x_train)
        print(y_pre.shape)
        # 优化器梯度清零
        optimizer.zero_grad()
        # 计算损失值
        loss = lossfunction(y_pre, y_train)
        # 后向传播
        loss.backward()
        # 模型更新参数
        optimizer.step()
        # 如果loss比前一次大就停止
        if temp > loss.item():
            sum_loss += loss.item()
            temp = loss.item()
            # 记录每次loss值
            loss_y[epoch] = loss.item()
        else:
            break
        print("loss=", loss.item())
        print("epoch = ", epoch+1)
    avg_loss = sum_loss/epochs
    print("avg_loss=", avg_loss)
    return loss_y


if __name__ == "__main__":
    x = torch.linspace(0, 999, 1000)
    y = train()

    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=x,
            y=y,
        )
    )
    fig.show()
