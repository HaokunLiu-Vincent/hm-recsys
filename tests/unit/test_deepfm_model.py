"""Tests for deepfm.model and deepfm.loss: the FM term, the components, the two losses."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from hmrec.config import DeepFMModelConfig, load_deepfm_config
from hmrec.deepfm.inputs import FieldLayout
from hmrec.deepfm.loss import listwise_loss, pointwise_loss
from hmrec.deepfm.model import DeepFM, fm_second_order

REPO = Path(__file__).resolve().parents[2]
REAL = load_deepfm_config(REPO / "configs" / "deepfm.yaml").model
SMALL = replace(REAL, embedding_dim=2, hidden_units=(3,))
LAYOUT = FieldLayout((3, 2), 0)  # field 0 has codes 0..2, field 1 has codes 0..1


def _model(config: DeepFMModelConfig = SMALL, layout: FieldLayout = LAYOUT) -> DeepFM:
    torch.manual_seed(0)
    return DeepFM(layout, config)


def test_fm_second_order_is_the_sum_of_all_pairwise_dot_products() -> None:
    # Three fields with embeddings (1, 2), (3, 4), (-1, 0.5):
    #   <e1, e2> = 3 + 8 = 11,  <e1, e3> = -1 + 1 = 0,  <e2, e3> = -3 + 2 = -1  -> 10.
    row = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]]])
    assert fm_second_order(row).tolist() == [10.0]
    # A single field has no pair.
    assert fm_second_order(row[:, :1]).tolist() == [0.0]
    # Each row on its own.
    two = torch.cat([row, 2 * row])
    assert fm_second_order(two).tolist() == [10.0, 40.0]


def test_default_model_has_the_shapes_of_the_proposal() -> None:
    layout = FieldLayout((33,) * 63 + (5,) * 8, 0)
    model = _model(REAL, layout)
    assert model.embedding.weight.shape == (63 * 33 + 8 * 5, 16)
    # The deep part: 71 fields x 16 = 1,136 -> 256 -> 128 -> 64 -> 1.
    linear = [layer for layer in model.mlp if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [
        (1136, 256),
        (256, 128),
        (128, 64),
        (64, 1),
    ]
    codes = torch.zeros((5, 71), dtype=torch.int64)
    assert model(codes).shape == (5,)
    # First-order weights and the bias start at 0; embeddings are small.
    assert float(model.first_order.weight.detach().abs().sum()) == 0.0
    assert model.bias.item() == 0.0
    assert 0.005 < float(model.embedding.weight.detach().std()) < 0.015


def test_each_field_reads_its_own_rows_of_the_tables() -> None:
    model = _model()
    with torch.no_grad():
        model.first_order.weight.copy_(torch.tensor([[1.0], [2.0], [4.0], [8.0], [16.0]]))
        model.bias.fill_(0.5)
    codes = torch.tensor([[0, 0], [2, 1], [1, 0]])
    # Field 0 uses rows 0..2, field 1 rows 3..4: 0.5 + w[c0] + w[3 + c1].
    assert model.first_order_logit(codes, None, None).tolist() == [9.5, 20.5, 10.5]
    embeddings = model.field_embeddings(codes, None, None)
    assert embeddings.shape == (3, 2, 2)
    assert torch.equal(embeddings[1, 0], model.embedding.weight[2])
    assert torch.equal(embeddings[1, 1], model.embedding.weight[4])


def _variant_model(components: str) -> DeepFM:
    """A model of one component variant with visible first- and second-order terms."""
    model = _model(replace(SMALL, components=components))
    with torch.no_grad():
        model.first_order.weight.copy_(torch.tensor([[1.0], [2.0], [4.0], [8.0], [16.0]]))
        model.embedding.weight.mul_(100.0)  # make the second-order term visible
    return model


def test_the_four_component_variants_add_up() -> None:
    codes = torch.tensor([[0, 1], [2, 0], [1, 1]])
    # The same seed gives the same weights in every variant, so the parts can be compared.
    outputs = {
        components: _variant_model(components)(codes).detach()
        for components in ("first_order", "fm", "deep", "full")
    }
    full = _variant_model("full")
    first = full.first_order_logit(codes, None, None).detach()
    second = fm_second_order(full.field_embeddings(codes, None, None)).detach()
    assert torch.allclose(outputs["first_order"], first)
    assert torch.allclose(outputs["fm"], first + second)
    assert torch.allclose(outputs["full"], outputs["fm"] + outputs["deep"])
    assert not torch.allclose(outputs["fm"], outputs["first_order"])
    assert float(outputs["deep"].abs().sum()) > 0


def test_first_order_variant_never_looks_up_an_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    # The first_order variant uses no embedding, so it must not compute them at all.
    codes = torch.tensor([[0, 1], [2, 0]])

    def refuse(*args: object) -> torch.Tensor:
        raise AssertionError("field_embeddings was called")

    first_order = _model(replace(SMALL, components="first_order"))
    expected = first_order.first_order_logit(codes, None, None)
    monkeypatch.setattr(first_order, "field_embeddings", refuse)
    assert torch.equal(first_order(codes), expected)
    # The other variants do need them.
    full = _model(replace(SMALL, components="full"))
    monkeypatch.setattr(full, "field_embeddings", refuse)
    with pytest.raises(AssertionError, match="field_embeddings was called"):
        full(codes)


def test_value_fields_use_the_value_and_the_missing_indicator() -> None:
    layout = FieldLayout((3,), 2)  # one code field, two value fields
    model = _model(layout=layout)
    with torch.no_grad():
        model.value_weights.copy_(torch.tensor([2.0, 3.0]))
        model.missing_weights.copy_(torch.tensor([10.0, 20.0]))
    codes = torch.tensor([[1], [1]])
    values = torch.tensor([[0.5, 0.0], [-1.0, 2.0]])
    missing = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    # First order: w . z + c . m  ->  1 + 20 = 21 and -2 + 6 = 4.
    assert model.first_order_logit(codes, values, missing).tolist() == pytest.approx([21.0, 4.0])
    embeddings = model.field_embeddings(codes, values, missing)
    assert embeddings.shape == (2, 3, 2)
    # e = z * v + m * u: a missing value contributes only the "missing" vector.
    assert torch.allclose(embeddings[0, 1], 0.5 * model.value_vectors[0])
    assert torch.allclose(embeddings[0, 2], model.missing_vectors[1])
    assert model(codes, values, missing).shape == (2,)
    with pytest.raises(ValueError, match="values and missing are needed"):
        model(codes)


def test_dropout_is_only_active_in_training() -> None:
    model = _model(replace(SMALL, dropout=0.5, hidden_units=(64,)))
    codes = torch.tensor([[0, 1]] * 8)
    model.eval()
    assert torch.equal(model(codes), model(codes))
    model.train()
    torch.manual_seed(1)
    first = model(codes)
    assert not torch.equal(first, model(codes))


# ---------------------------------------------------------------- losses


def test_pointwise_loss_hand_computed() -> None:
    # logit 0 -> p = 0.5 -> -ln 0.5 for either label; logit ln 3 -> p = 0.75.
    logits = torch.tensor([0.0, 0.0, math.log(3.0), math.log(3.0)])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    expected = (2 * math.log(2) - math.log(0.75) - math.log(0.25)) / 4
    assert float(pointwise_loss(logits, labels)) == pytest.approx(expected)


def test_listwise_loss_hand_computed() -> None:
    # Query 0: logits (0, 0), first row positive -> softmax 1/2 -> loss ln 2.
    # Query 1: logits (ln 3, 0, 0), rows 0 and 2 positive -> softmax (3/5, 1/5, 1/5)
    #          -> loss -(ln 3/5 + ln 1/5) / 2.
    logits = torch.tensor([0.0, 0.0, math.log(3.0), 0.0, 0.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    query_index = torch.tensor([0, 0, 1, 1, 1])
    query_0 = math.log(2)
    query_1 = -(math.log(3 / 5) + math.log(1 / 5)) / 2
    loss = listwise_loss(logits, labels, query_index, 2)
    assert float(loss) == pytest.approx((query_0 + query_1) / 2)


def test_listwise_loss_does_not_depend_on_row_order_or_on_a_shift() -> None:
    logits = torch.tensor([0.3, -1.2, 2.0, 0.1, 0.7])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    query_index = torch.tensor([0, 0, 1, 1, 1])
    base = listwise_loss(logits, labels, query_index, 2)
    order = torch.tensor([3, 0, 4, 1, 2])
    shuffled = listwise_loss(logits[order], labels[order], query_index[order], 2)
    assert float(shuffled) == pytest.approx(float(base))
    # Adding a constant to all logits of a query does not change its softmax; very
    # large logits must not overflow.
    shifted = listwise_loss(logits + 1000.0, labels, query_index, 2)
    assert float(shifted) == pytest.approx(float(base), abs=1e-4)


def test_listwise_loss_needs_a_positive_in_every_query() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0])
    labels = torch.tensor([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="no positive row"):
        listwise_loss(logits, labels, torch.tensor([0, 0, 1]), 2)
