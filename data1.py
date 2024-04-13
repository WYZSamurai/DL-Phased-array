import torch
import plotly.graph_objects as go


def generate(batch_size: int, scale: int, theta_max: float, theta_min: float):
    delta = int(scale*(theta_max-theta_min))
    theta = torch.linspace(theta_min, theta_max, delta)
    # (batch_size,)[40,140]
    direction = scale*torch.randint(40, 141, (batch_size,))
    # (batch_size,)[-30,-20]
    sll = torch.randint(-30, -19, (batch_size,))
    print(direction)
    print(sll)
    # (batch_size,)[20,30]
    width = -sll*scale
    print(width)
    Fdb = torch.ones(batch_size, delta) * \
        (sll.reshape(batch_size, 1).repeat(1, delta))
    for i in range(batch_size):
        Fdb[i, direction[i]-width[i]:direction[i]+width[i]] = 0
    return Fdb, theta


def plot(theta: torch.Tensor, Fdb: torch.Tensor, theta_max: float, theta_min: float):
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=theta,
            y=Fdb,
        )
    )
    fig.update_layout(
        template="simple_white",
        title="方向图",
        xaxis_title="theta",
        yaxis_title="Fdb",
        xaxis_range=[theta_min-10, theta_max+10],
        yaxis_range=[-60, 0.5],
    )
    fig.show()
