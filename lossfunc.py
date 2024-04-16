import torch


# 计算(Batch_size,delta)的方向图副瓣电平(Batch_size,)
def msll(Fdb: torch.Tensor, mask: torch.Tensor):
    temp = Fdb.clone()

    temp[mask] = float('-inf')
    MSLL = temp.max(dim=1).values
    return MSLL


# 计算总损失值(主瓣MSE和MSLL)
def total_loss(transformed_outputs: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor):
    lossfun = torch.nn.MSELoss(reduction="mean")
    # (batch_size,)
    loss_mll: torch.Tensor = lossfun(transformed_outputs[mask], inputs[mask])

    MSLL = msll(transformed_outputs, mask)
    loss_msll: torch.Tensor = lossfun(MSLL, torch.zeros_like(MSLL)-50)

    (w1, w2) = (0.5, 0.5)

    return w1*loss_mll+w2*loss_msll
