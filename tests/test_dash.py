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

    optim.shrink_params()

    optim.clear_grad_ema()

def test_shrink_with_dataset():
    from torch.utils.data import Dataset
    from DASH.DASH import AdamW, shrink_params_with_dataset_

    from torch.nn import Linear, Sequential
    from einops.layers.torch import Reduce

    class MockDataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.randn(10)

    net = Sequential(Linear(10, 1), Reduce('... ->', 'sum'))

    dataset = MockDataset()

    shrink_params_with_dataset_(net, dataset)
