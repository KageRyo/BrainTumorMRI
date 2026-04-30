from __future__ import annotations

from brain_tumor_mri.runtime import build_model


def test_build_model_uses_config_model_section() -> None:
    model = build_model(
        {
            "model": {
                "backbone": "convnext_tiny",
                "pretrained": False,
                "num_classes": 4,
                "in_channels": 1,
            }
        }
    )

    assert model.classifier[-1].out_features == 4
