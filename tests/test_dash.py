import torch
import pytest

def test_dash():
    from DASH.DASH import AdamW
    from torch.nn import Linear

    net = Linear(10, 5)
    optim = AdamW(net.parameters(), lr = 3e-4)

    loss = net(torch.randn(10)).sum()
    loss.backward()

    optim.step()
    optim.zero_grad()
