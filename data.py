import torch
from sklearn.model_selection import train_test_split


# 建立数据集
def generator(batch_size: int):
    x = torch.zeros(size=(batch_size, 181), dtype=torch.float64)
    y = torch.zeros(size=(batch_size, 48), dtype=torch.float64)
    for i in range(batch_size):
        temp_x = 100 * torch.rand(size=(181,), dtype=torch.float64)
        temp_y = torch.rand(size=(48,), dtype=torch.float64)
        x[i] = temp_x
        y[i] = temp_y
    return x, y


# 分割数据集
def spl_data(batch_size: int):
    x, y = generator(batch_size=batch_size)
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return x_train, x_test, y_train, y_test


# 输入数据
def inputdata(batch_size=4):
    # (n,1)
    direction = torch.randint(40, 141, size=(batch_size,))
    # (n,1)
    sll = torch.randint(21, 30, size=(batch_size, 1))
    # (n,181)
    perceptual = -1 * sll * torch.ones(size=(batch_size, 181))
    for i in range(batch_size):
        width = sll[i]
        perceptual[i, direction[i]-1-width:direction[i]+width] = 0
    return direction, perceptual


# n元阵方向图叠加
# mag(n,181) ang(n,181)
def arg(mag: torch.Tensor, ang: torch.Tensor, element_num=24,):
    # 方向图乘积
    E = torch.zeros(size=(181,), dtype=torch.float64)
    for i in range(element_num):
        E = E+mag[i]*torch.exp(1j*((i+1)*torch.pi+(ang[i]/180)))
    E = torch.abs(E)
    return E


if __name__ == "__main__":
    pass
