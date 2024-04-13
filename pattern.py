import plotly.graph_objects as go
import torch


# 线阵，等间距，等幅同相
# dna(ME,) Fdb(delta,)
def pattern(mag: torch.Tensor, phase_0: torch.Tensor, lamb: float, d: float, delta: int, theta_0: float):
    m = mag.shape[0]
    pi = torch.pi
    k = 2*pi/lamb

    theta_0 = torch.tensor(theta_0)*pi/180
    theta = torch.linspace(-pi/2, pi/2, delta)

    # phi(delta,)
    phi = (torch.sin(theta)-torch.sin(theta_0))
    # nd(m,)
    dm = k*d*torch.arange(0, m)

    F = torch.zeros(delta,)
    for i in range(m):
        phase = phase_0[i]+dm[i]*phi
        F = F+mag[i]*torch.exp(torch.complex(torch.zeros_like(phase), phase))
    F = F.abs()
    Fdb = 20*torch.log10(F/F.max())
    # print("Fdb为：\n", Fdb)
    return Fdb


def plot(Fdb: torch.Tensor, delta: int, theta_min: float, theta_max: float):
    theta = torch.linspace(theta_min, theta_max, delta)
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


if __name__ == "__main__":
    lamb = 1.0
    d = 0.5*lamb
    theta_0 = 0.0
    delta = 360

    m = 10
    mag = torch.randint(0, 2, (m,))
    phase_0 = torch.zeros(m,)

    Fdb = pattern(mag, phase_0, lamb, d, delta, theta_0)
    plot(Fdb, delta, -90.0, 90.0)
