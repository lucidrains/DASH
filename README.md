<img src="./dash-fig4.png" width="300px"></img>

## DASH (wip)

Implementation of DASH, [Warm-Starting Neural Network Training in Stationary Settings without Loss of Plasticity](https://arxiv.org/abs/2410.23495)

## Install

```bash
$ pip install DASH-pytorch
```

## Usage

```python
import torch
from torch.nn import Linear

from DASH.DASH import AdamW

net = Linear(10, 5)
optim = AdamW(net.parameters(), lr = 3e-4)

loss = net(torch.randn(10)).sum()
loss.backward()

optim.step()
optim.zero_grad()

optim.shrink_params()
optim.clear_grad_ema()
```

## Citations

```bibtex
@misc{shin2024dashwarmstartingneuralnetwork,
    title   = {DASH: Warm-Starting Neural Network Training in Stationary Settings without Loss of Plasticity}, 
    author  = {Baekrok Shin and Junsoo Oh and Hanseul Cho and Chulhee Yun},
    year    = {2024},
    eprint  = {2410.23495},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2410.23495}, 
}
```
