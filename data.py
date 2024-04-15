import torch
import plotly.graph_objects as go


# 生成理想的方向图
def generate(batch_size: int, delta: int):
    scale = int(delta/180)
    direction = scale*torch.randint(40, 141, (batch_size,))
    sll = -1*(torch.randint(20, 30, size=(batch_size,),
              dtype=(torch.float))+torch.rand(batch_size,))
    width = (-sll/2).to(dtype=torch.int)
    # Fdb(batch_size, delta)
    Fdb = torch.ones(batch_size, delta) * sll.unsqueeze(1)

    # 生成mask矩阵
    indexes = torch.arange(delta).unsqueeze(0).repeat(
        batch_size, 1)  # 生成0到delta-1的索引并复制成(batch_size, delta)形状
    lower_bounds = (direction - width).unsqueeze(1)
    upper_bounds = (direction + width).unsqueeze(1)
    # 主瓣位置
    # mask(batch_size, delta)
    mask = (indexes >= lower_bounds) & (indexes <= upper_bounds)  # 创建布尔掩码

    Fdb[mask] = 0  # 使用掩码将指定位置置0
    return Fdb, mask


# 根据相位和激励计算线阵方向图
def pattern(mag: torch.Tensor, phase_0: torch.Tensor, lamb: float, d: float, delta: int, theta_0: float):
    # mag/phase_0(batch_size,m)
    m = mag.shape[1]
    pi = torch.pi
    k = 2 * pi / lamb

    theta_0_rad = torch.tensor(theta_0) * pi / 180
    theta = torch.linspace(-pi / 2, pi / 2, delta,
                           device=mag.device)  # 生成theta并指定设备

    # phi(delta,)
    phi = (torch.sin(theta) - torch.sin(theta_0_rad)
           ).unsqueeze(0).to(mag.device)  # 增加一个维度，以便与dm进行广播

    # nd(m,)
    # 将向量变为列向量进行广播
    dm = k * d * torch.arange(0, m, device=mag.device).unsqueeze(1)

    # 计算每个批次所有m值和所有delta值的所有相位
    # 广播以创建相位值的（batch_size，m，delta）矩阵
    phase = phase_0.unsqueeze(2) + dm * phi

    # 使用欧拉公式将相位转换为复数并求和
    # 广播mag到（batch_size，m，delta）
    complex_exponential = mag.unsqueeze(
        2) * torch.exp(torch.complex(torch.zeros_like(phase), phase))
    # 沿m维度求和，取大小
    F = torch.sum(complex_exponential, dim=1).abs()

    # 转换为db，按批次中每个个体的最大值进行归一化
    Fdb = 20 * torch.log10(F / F.max(dim=1, keepdim=True).values)

    return Fdb


# 绘图
def plot(Fdb: torch.Tensor):
    delta = Fdb.shape[0]
    theta = torch.linspace(-90.0, 90.0, delta)
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=theta,
            y=Fdb.to(torch.device("cpu")),
        )
    )
    fig.update_layout(
        template="simple_white",
        title="方向图",
        xaxis_title="theta",
        yaxis_title="Fdb",
        xaxis_range=[-100.0, 100.0],
        yaxis_range=[-60, 0.5],
    )
    fig.show()
