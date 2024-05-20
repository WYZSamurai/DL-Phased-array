import torch


def total_loss(transformed_outputs: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor):
    """
    计算总损失值(主瓣MSE和MSLL)
    """
    mse_loss = torch.nn.MSELoss(reduction="mean")
    # (batch_size,)
    loss_mll: torch.Tensor = mse_loss(transformed_outputs[mask], inputs[mask])

    l1_loss = torch.nn.L1Loss()
    diff = torch.relu(transformed_outputs[~mask]-inputs[~mask])
    loss_sll: torch.Tensor = l1_loss(
        torch.zeros_like(diff, device=diff.device), diff)

    (w1, w2) = (0.10, 0.05)

    return w1*loss_mll+w2*loss_sll
