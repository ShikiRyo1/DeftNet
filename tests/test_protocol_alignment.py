from pathlib import Path

import torch

from deftnet.losses import CombinedSegLoss
from deftnet.models import DeftNet, DeftNetConfig, ExpertSegmentor, build_encoder
from deftnet.training import assign_perspective_folds


def test_canonical_expert_roster_and_admission_policy():
    cfg = DeftNetConfig()
    assert cfg.expert_names == ("E1", "E2", "E3", "E4", "E5")
    assert cfg.band_policy["e1"] == ("E1", "E2", "E3")
    assert cfg.band_policy["e3"] == cfg.expert_names
    assert cfg.band_policy["e5"] == ("E4", "E5")


def test_legacy_roster_is_migrated():
    cfg = DeftNetConfig(
        expert_names=("E5", "E7", "E9", "E11", "E12"),
        band_policy={
            "e1": ("E5", "E7", "E9"),
            "e2": ("E5", "E7", "E9"),
            "e3": ("E5", "E7", "E9", "E11", "E12"),
            "e4": ("E11", "E12"),
            "e5": ("E11", "E12"),
        },
    )
    assert cfg.expert_names == ("E1", "E2", "E3", "E4", "E5")
    assert cfg.band_policy["e4"] == ("E4", "E5")


def test_current_loss_defaults_to_equal_bce_and_dice():
    loss = CombinedSegLoss()
    assert loss.w_bce == 0.5
    assert loss.w_dice == 0.5
    assert loss.w_cldice == 0.0


def test_perspective_folds_are_deterministic_and_balanced():
    sample_ids = [f"case_{index:04d}" for index in range(1408)]
    first = assign_perspective_folds(sample_ids, n_folds=5, seed=2026)
    second = assign_perspective_folds(list(reversed(sample_ids)), n_folds=5, seed=2026)
    assert first == second
    sizes = [sum(fold == index for fold in first.values()) for index in range(5)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 1408


def test_phase1_encoder_checkpoint_load(tmp_path: Path):
    cfg = DeftNetConfig(base_channels=2, use_feature_adapters=False, hsaf_gate_dropout=0.0)
    checkpoints = {}
    for name in cfg.expert_names:
        expert = ExpertSegmentor(name, cfg)
        path = tmp_path / f"{name}.pth"
        torch.save({"encoder_state_dict": expert.encoder.state_dict()}, path)
        checkpoints[name] = path

    model = DeftNet(cfg)
    resolved = model.load_expert_checkpoints(checkpoints, strict=True)
    assert set(resolved) == set(cfg.expert_names)
    assert all(not parameter.requires_grad for encoder in model.encoders.values() for parameter in encoder.parameters())


def test_hsaf_masks_disallowed_experts():
    cfg = DeftNetConfig(base_channels=2, hsaf_gate_dropout=0.0, use_feature_adapters=False)
    model = DeftNet(cfg).eval()
    with torch.no_grad():
        model(torch.randn(1, 1, 32, 32))
    shallow_weights = model.hsaf["e1"].last_weights
    deep_weights = model.hsaf["e5"].last_weights
    assert shallow_weights is not None and deep_weights is not None
    assert torch.count_nonzero(shallow_weights[:, 3:]) == 0
    assert torch.count_nonzero(deep_weights[:, :3]) == 0


def test_parameter_accounting_matches_released_graph():
    model = DeftNet(DeftNetConfig())
    assert model.trainable_parameters() == 1_862_074
    assert model.total_parameters() == 12_364_598


def test_expert_parameter_counts_match_architecture_specification():
    cfg = DeftNetConfig()
    expected = {
        "E1": 4_326_096,
        "E2": 1_651_472,
        "E3": 135_740,
        "E4": 2_171_728,
        "E5": 2_217_488,
    }
    observed = {
        name: sum(parameter.numel() for parameter in build_encoder(name, cfg).parameters())
        for name in cfg.expert_names
    }
    assert observed == expected
