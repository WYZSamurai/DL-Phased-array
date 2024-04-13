import torch
import plotly.graph_objects as go
import torch
import csv


# 输入数据
def inputdata(batch_size=4):
    direction = torch.randint(40, 141, size=(batch_size,))
    sll = torch.randint(21, 30, size=(batch_size, 1))
    perceptual = -1 * sll * torch.ones(size=(batch_size, 181))
    for i in range(batch_size):
        width = sll[i]
        perceptual[i, direction[i]-1-width:direction[i]+width] = 0
    return direction, perceptual


def read(path="", skip=1, col=2):
    # 设置文件路径
    # 设置跳过的行数
    # 设置读取的列数
    data = []
    # 按行扫描，每列都循环一次，共col次
    for i in range(col):
        with open(path, mode="r", encoding="utf-8") as csvfile:
            temp = []
            cout = -skip
            # 设置扫描器
            reader = csv.reader(csvfile)
            # d为逐行扫描的数据
            for d in reader:
                if cout < 0:
                    cout += 1
                    continue
                temp.append(float(d[i]))
            data.append(temp)
    return data


# n元阵方向图叠加
def arg(element_num=24):
    # 设置相位
    ang = torch.zeros(size=(element_num, 181), dtype=torch.float64)
    for i in range(element_num):
        a = read(path="24_hori/ang_deg"+str(i+1)+".csv", skip=1, col=2)
        ang[i] = torch.tensor(a[1])
    # 设置幅度
    mag = torch.zeros(size=(element_num, 181), dtype=torch.float64)
    for i in range(element_num):
        a = read(path="24_hori/mag"+str(i+1)+".csv", skip=1, col=2)
        mag[i] = torch.tensor(a[1])
    # 方向图乘积
    E = torch.zeros(size=(181,), dtype=torch.float64)
    for i in range(element_num):
        E = E+mag[i]*torch.exp(1j*((i+1)*torch.pi/2+(ang[i]/180)))
    E = torch.abs(E)
    return E


if __name__ == "__main__":
    y = arg(24)
    x = torch.linspace(0, 180, 181)

    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=x,
            y=y,
        )
    )
    fig.show()
