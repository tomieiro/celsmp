import math

import numpy as np
import pytest

import celsmp
from celsmp import codec
from celsmp.cli import main

torch = pytest.importorskip("torch")


def sections(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "weights": {"embed": rng.normal(0, 0.02, (70, 50)).astype(np.float32),
                    "conv": rng.normal(0, 1, (16, 9, 5)).astype(np.float32),
                    "bias": rng.normal(0, 1, 50).astype(np.float32),
                    "steps": np.arange(6, dtype=np.int64), "mask": np.array([True, False])},
        "state": {"nested": {"t": np.ones((3, 3), np.float16), "rate": float("nan"), "n": 2 ** 60},
                  "by_id": {0: {"m": np.zeros(4, np.float64)}, 7: (1, "a", None)},
                  "events": [{"event": "stage", "ok": True}], "empty": np.zeros((0, 4), np.float32)},
    }


def assert_same(a, b):
    assert type(a) is type(b) or (isinstance(a, np.ndarray) and isinstance(b, np.ndarray))
    if isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    else:
        assert a == b


@pytest.mark.parametrize("name", ["raw", "zlib"])
def test_lossless_roundtrip_is_exact(tmp_path, name):
    original = sections()
    header = celsmp.save(tmp_path / "m.csmp", original, codec=name)
    loaded = celsmp.load(tmp_path / "m.csmp")
    assert_same(original, loaded.sections)
    assert header["compression"]["lossy"] is False
    assert all(t["offset"] % 64 == 0 for t in header["tensors"])


def test_header_makes_training_facts_explicit(tmp_path):
    celsmp.save(tmp_path / "m.csmp", sections(), model={"name": "toy", "betas": (0.9, 0.95)},
                training={"sft": True, "step": 10},
                metrics={"train_loss": 2.0, "val_loss": 3.0, "val_perplexity": 21.0, "accuracy": 0.5})
    header = celsmp.read_header(tmp_path / "m.csmp")
    assert header["training"] == {"sft": True, "step": 10, "regime": "sft"}
    assert header["metrics"] == {"train_loss": 2.0, "train_perplexity": math.exp(2.0), "val_loss": 3.0,
                                 "val_perplexity": 21.0, "accuracy": 0.5}
    assert header["model"]["betas"] == (0.9, 0.95)


def test_defaults_are_not_sft_not_lossy_and_metrics_unknown(tmp_path):
    header = celsmp.save(tmp_path / "m.csmp", sections())
    assert header["training"] == {"sft": False, "regime": "pretrain"}
    assert set(header["metrics"].values()) == {None}
    assert header["compression"]["codec"] == "raw"
    with pytest.raises(TypeError):
        celsmp.save(tmp_path / "m.csmp", sections(), training={"sft": "yes"})
    with pytest.raises(ValueError):
        celsmp.save(tmp_path / "m.csmp", {"state": {}})


def test_dct_is_lossy_only_on_eligible_weights(tmp_path):
    original = sections()
    original["state"]["big"] = np.random.default_rng(1).normal(size=(32, 32)).astype(np.float32)
    header = celsmp.save(tmp_path / "m.csmp", original, codec="dct", quality=95)
    loaded = celsmp.load(tmp_path / "m.csmp")
    assert header["compression"]["lossy"] is True
    assert {t["codec"] for t in header["tensors"] if t["section"] == "state"} == {"zlib"}
    assert_same(original["state"], loaded.sections["state"])
    for key in ("bias", "steps", "mask"):
        assert np.array_equal(original["weights"][key], loaded.weights[key])
    for key in ("embed", "conv"):
        a, b = original["weights"][key], loaded.weights[key]
        assert a.dtype == b.dtype and a.shape == b.shape
        relative = np.linalg.norm(a - b) / np.linalg.norm(a)
        assert 0 < relative < 0.15
    assert 0 < header["compression"]["rel_rmse"] < 0.15
    with open(tmp_path / "m.csmp", "rb") as file:
        assert file.read(16)[12] & 1


def test_dct_error_falls_with_quality_and_survives_outliers(tmp_path):
    rng = np.random.default_rng(3)
    weight = rng.normal(0, 0.02, (64, 96)).astype(np.float32)
    weight[5, 7] = 40.0                                      # 2000 sigma
    errors = []
    for quality in (30, 75, 100):
        table = codec.quant_table(quality, "jpeg")
        payload, params = codec.encode_dct(weight, table)
        restored = codec.decode_dct(payload, params, table, weight.shape, weight.dtype)
        errors.append(float(np.abs(restored - weight).max()))
        assert abs(restored[5, 7] - 40.0) < 4.0
    assert errors[0] > errors[1] > errors[2]
    assert codec.quant_table(100) == [1] * 64 and codec.quant_table(50, "jpeg")[:3] == [16, 11, 10]
    assert not codec.dct_eligible(np.full((8, 8), np.nan, np.float32))


def test_corruption_is_detected(tmp_path):
    path = tmp_path / "m.csmp"
    celsmp.save(path, sections())
    celsmp.verify(path)
    data = bytearray(path.read_bytes())
    data[-40] ^= 0xFF                                      # inside the last non-empty tensor
    path.write_bytes(data)
    with pytest.raises(celsmp.CSMPError):
        celsmp.verify(path)
    data = bytearray(path.read_bytes())
    data[60] ^= 0xFF
    path.write_bytes(data)
    with pytest.raises(celsmp.CSMPError):
        celsmp.read_header(path)
    path.write_bytes(b"PK\x03\x04 not a csmp")
    with pytest.raises(celsmp.CSMPError):
        celsmp.read_header(path)


def test_partial_mmap_and_torch_loading(tmp_path):
    original = sections()
    celsmp.save(tmp_path / "m.csmp", original)
    only = celsmp.load(tmp_path / "m.csmp", sections=["weights"], mmap=True)
    assert list(only.sections) == ["weights"] and isinstance(only.weights["embed"], np.memmap)
    assert np.array_equal(only.weights["embed"], original["weights"]["embed"])
    tensors = celsmp.load(tmp_path / "m.csmp", as_torch=True).weights
    assert tensors["embed"].dtype == torch.float32 and tensors["mask"].dtype == torch.bool
    tensors["embed"] += 1                                     # writable, not a view of the file
    with pytest.raises(KeyError):
        celsmp.load(tmp_path / "m.csmp", sections=["nope"])


def civ_checkpoint():
    layer = torch.nn.Linear(16, 12)
    optimizer = torch.optim.AdamW(layer.parameters())
    layer(torch.randn(2, 16)).sum().backward()
    optimizer.step()
    return {"format": "cellm-civ/1", "model": layer.state_dict(), "optimizer": optimizer.state_dict(),
            "civilization": {"culture": {"keys": torch.randn(4, 8)}, "generation": 3, "events": []},
            "trainer": {"step": 7, "tokens": 900, "train_seconds": 1.5, "stage_index": 1},
            "config": {"training": {"stages": [{"name": "umg-baseline"}, {"name": "society"}]}}}


def test_cli_pack_info_verify_unpack(tmp_path, capsys):
    payload = civ_checkpoint()
    torch.save(payload, tmp_path / "ckpt.pt")
    (tmp_path / "metrics.jsonl").write_text('{"step": 6, "loss": 9.0}\n{"step": 7, "loss": 2.5, "ppl": 12.2}\n')
    (tmp_path / "eval.jsonl").write_text('{"step": 7, "label": "end-society", "val_loss": 2.75, "val_ppl": 15.6}\n')

    def run(*args):
        main([str(arg) for arg in args])
        return capsys.readouterr().out

    shown = run("pack", tmp_path / "ckpt.pt", "-o", tmp_path / "m.csmp", "--name", "toy", "--run-dir", tmp_path,
                "--with-optimizer", "--metric", "bleu=0.25")
    assert "SFT             no" in shown and "2.7500" in shown and "society" in shown
    assert "ok" in run("verify", tmp_path / "m.csmp")
    header = celsmp.read_header(tmp_path / "m.csmp")
    assert header["training"]["stage"] == "society" and header["training"]["step"] == 7
    assert header["metrics"]["train_loss"] == 2.5 and header["metrics"]["val_perplexity"] == 15.6
    assert header["metrics"]["bleu"] == 0.25 and header["model"]["parameters"] == 16 * 12 + 12
    assert list(header["sections"]) == ["weights", "civilization", "optimizer"]

    run("unpack", tmp_path / "m.csmp", "-o", tmp_path / "back.pt")
    back = torch.load(tmp_path / "back.pt", weights_only=True)
    assert back["format"] == "cellm-civ/1" and back["trainer"] == payload["trainer"]
    assert back["config"] == payload["config"]
    assert all(torch.equal(back["model"][k], v) for k, v in payload["model"].items())
    assert torch.equal(back["civilization"]["culture"]["keys"], payload["civilization"]["culture"]["keys"])
    state = back["optimizer"]["state"]
    assert list(state) == [0, 1] and torch.equal(state[0]["exp_avg"], payload["optimizer"]["state"][0]["exp_avg"])
    torch.optim.AdamW(torch.nn.Linear(16, 12).parameters()).load_state_dict(back["optimizer"])


def test_plain_state_dict_and_sft_flag(tmp_path):
    torch.save({"state_dict": torch.nn.Linear(4, 4).state_dict()}, tmp_path / "plain.pt")
    main(["pack", str(tmp_path / "plain.pt"), "--sft", "--val-loss", "1.0"])
    header = celsmp.read_header(tmp_path / "plain.csmp")
    assert header["training"]["sft"] is True and header["training"]["regime"] == "sft"
    assert header["metrics"]["val_perplexity"] == math.e and header["source"]["format"] == "torch-state-dict"


def test_library_pack_describe_unpack_without_the_cli(tmp_path):
    payload = civ_checkpoint()
    torch.save(payload, tmp_path / "ckpt.pt")
    path = celsmp.pack(tmp_path / "ckpt.pt", sft=True, metrics={"train_loss": 1.0, "val_loss": 2.0},
                       codec="dct", quality=100, notes="toy")
    assert path == tmp_path / "ckpt.csmp" and path.read_bytes()[:8] == b"\x89CSMP\r\n\x1a"
    header = celsmp.read_header(path)
    assert header["format"] == "csmp" and header["training"]["sft"] is True
    assert len(header["source"]["sha256"]) == 64
    text = celsmp.describe(path, tensors=True)
    assert "SFT             yes" in text and "LOSSY" in text and "2.0000" in text
    back = torch.load(celsmp.unpack(path, tmp_path / "back.pt"), weights_only=True)
    assert torch.allclose(back["model"]["weight"], payload["model"]["weight"], atol=1e-2)
    assert torch.equal(back["model"]["bias"], payload["model"]["bias"])
    with pytest.raises(FileNotFoundError):
        celsmp.pack(tmp_path / "missing.pt")


def test_describe_method_shows_model_data(tmp_path):
    header = celsmp.save(tmp_path / "m.csmp", sections(), metrics={"val_loss": 2.0},
                         model={"name": "toy", "parameters": 12, "config": {"core": {"d": 64, "steps": 4}, "seed": 1}})
    loaded = celsmp.load(tmp_path / "m.csmp", sections=["weights"])
    text = loaded.describe(config=True, tensors=True)
    assert text == celsmp.describe(tmp_path / "m.csmp", tensors=True, config=True) and str(loaded) == loaded.describe()
    assert "toy 12 parameters" in text and "[config]" in text and "core.d" in text and "float32" in text
    assert "[config]" not in str(loaded)
    assert "toy 12 parameters" in celsmp.describe(header)          # a header is enough, no file needed
