import torch

from deftnet.models import DeftNet, DeftNetConfig


def test_deftnet_forward_shape():
    cfg = DeftNetConfig(base_channels=4, use_feature_adapters=False, hsaf_gate_dropout=0.0)
    model = DeftNet(cfg)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 1, 64, 64))
    assert tuple(y.shape) == (1, 1, 64, 64)
