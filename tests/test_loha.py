# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
from torch import nn

from peft import LoHaConfig, LoKrConfig, get_peft_model
from peft.tuners.loha.layer import HadaWeight, HadaWeightCP


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float64, requires_grad=True)


class TestLoHaCustomGradients:
    # HadaWeight and HadaWeightCP implement their backward pass manually, check it against numerical gradients

    def test_hada_weight_gradcheck(self):
        torch.manual_seed(0)
        r, out_features, in_features = 3, 5, 4
        args = (_rand(out_features, r), _rand(r, in_features), _rand(out_features, r), _rand(r, in_features))
        scale = torch.tensor(0.7, dtype=torch.float64)
        assert torch.autograd.gradcheck(lambda *tensors: HadaWeight.apply(*tensors, scale), args)

    # (3, 1) is also the layout used for Conv1d layers
    @pytest.mark.parametrize("kernel_size", [(3, 3), (3, 1), (1, 3)])
    def test_hada_weight_cp_gradcheck(self, kernel_size):
        torch.manual_seed(0)
        r, out_channels, in_channels = 3, 5, 4
        args = (
            _rand(r, r, *kernel_size),
            _rand(r, out_channels),
            _rand(r, in_channels),
            _rand(r, r, *kernel_size),
            _rand(r, out_channels),
            _rand(r, in_channels),
        )
        scale = torch.tensor(0.7, dtype=torch.float64)
        assert torch.autograd.gradcheck(lambda *tensors: HadaWeightCP.apply(*tensors, scale), args)


class _LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.lin(x)


def _patch_rand(monkeypatch, draws):
    pending = iter(draws)

    def fake_rand(*args, **kwargs):
        return next(pending).clone()

    monkeypatch.setattr(torch, "rand", fake_rand)


class TestRankDropoutEmptyMask:
    # Rank dropout rescales the keep-mask by its mean. If every output row is dropped, that mean is 0 and the
    # delta weight becomes NaN. An empty mask should instead contribute a zero delta for the step.

    @pytest.mark.parametrize(
        "config_cls,config_kwargs,out_features",
        [
            (LoHaConfig, {"rank_dropout": 0.99}, 4),
            (LoHaConfig, {"rank_dropout": 0.5}, 1),
            (LoKrConfig, {"rank_dropout": 0.99, "rank_dropout_scale": True}, 4),
            (LoKrConfig, {"rank_dropout": 0.5, "rank_dropout_scale": True}, 1),
            (LoKrConfig, {"rank_dropout": 0.99, "rank_dropout_scale": False}, 4),
        ],
        ids=["loha-out4", "loha-out1", "lokr-scale-out4", "lokr-scale-out1", "lokr-noscale-out4"],
    )
    def test_all_zero_mask_forward_matches_base(self, monkeypatch, config_cls, config_kwargs, out_features):
        torch.manual_seed(0)
        in_features = 4 if out_features != 1 else 1
        model = _LinearModel(in_features, out_features)
        x = torch.randn(2, in_features)
        with torch.no_grad():
            base_out = model(x).clone()

        rank = 1 if out_features == 1 else 2
        config = config_cls(target_modules=["lin"], r=rank, init_weights=False, **config_kwargs)
        peft_model = get_peft_model(model, config)
        peft_model.train()

        # Forward draws torch.rand(1) for module dropout, then one value per output row for the rank mask.
        # Returning 1 keeps the adapter enabled so the empty rank mask is actually applied.
        _patch_rand(monkeypatch, [torch.ones(1), torch.zeros(out_features)])
        output = peft_model(x)
        assert torch.isfinite(output).all()
        assert torch.equal(output, base_out)

    @pytest.mark.parametrize(
        "config_cls,scale",
        [(LoHaConfig, True), (LoKrConfig, True), (LoKrConfig, False)],
        ids=["loha", "lokr-scale", "lokr-noscale"],
    )
    def test_partial_mask_keeps_existing_scaling(self, monkeypatch, config_cls, scale):
        # A mask that still keeps some rows must be unchanged: LoHa always divides by the keep-rate, LoKr only
        # when rank_dropout_scale is set.
        torch.manual_seed(0)
        model = _LinearModel(4, 4)
        kwargs = {"rank_dropout": 0.5}
        if config_cls is LoKrConfig:
            kwargs["rank_dropout_scale"] = scale
        config = config_cls(target_modules=["lin"], r=2, init_weights=False, **kwargs)
        peft_model = get_peft_model(model, config)
        layer = peft_model.base_model.model.lin
        unscaled = layer.get_delta_weight("default", apply_rank_dropout=False)

        samples = torch.tensor([0.1, 0.9, 0.4, 0.6])
        _patch_rand(monkeypatch, [samples])
        scaled = layer.get_delta_weight("default", apply_rank_dropout=True)

        keep = (samples > 0.5).to(dtype=unscaled.dtype).view(-1, 1)
        if scale:
            keep = keep / keep.mean()
        expected = unscaled * keep
        assert torch.isfinite(scaled).all()
        assert torch.equal(scaled, expected)

    def test_rank_dropout_one_is_ignored_in_eval(self):
        # rank_dropout=1 drops every row in train mode (torch.rand is in [0, 1)). Eval must still apply the full delta.
        torch.manual_seed(0)
        model = _LinearModel(4, 4)
        x = torch.randn(2, 4)
        config = LoHaConfig(target_modules=["lin"], r=2, init_weights=False, rank_dropout=1.0)
        peft_model = get_peft_model(model, config)
        layer = peft_model.base_model.model.lin
        base = layer.get_base_layer()

        peft_model.eval()
        eval_out = peft_model(x)
        with torch.no_grad():
            delta = layer.get_delta_weight("default", apply_rank_dropout=False)
            full = nn.functional.linear(x, base.weight + delta, base.bias)
            base_out = nn.functional.linear(x, base.weight, base.bias)
        assert torch.isfinite(eval_out).all()
        torch.testing.assert_close(eval_out, full)
        assert not torch.equal(eval_out, base_out)

        peft_model.train()
        train_out = peft_model(x)
        assert torch.isfinite(train_out).all()
        assert torch.equal(train_out, base_out)
