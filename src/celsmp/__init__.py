"""CelSMP: the ``.csmp`` model storage format.

    import celsmp

    celsmp.save("model.csmp", {"weights": model.state_dict()},
                training={"sft": False}, metrics={"train_loss": 3.9, "val_loss": 4.4})
    celsmp.read_header("model.csmp")["metrics"]          # no tensor is read
    model.load_state_dict(celsmp.load("model.csmp", as_torch=True).weights)
"""

from celsmp.checkpoint import from_checkpoint, pack, run_metrics, to_checkpoint, unpack
from celsmp.container import CSMP, CSMPError, load, read_header, save, verify
from celsmp.report import describe

__version__ = "0.2.0"
__all__ = ["CSMP", "CSMPError", "describe", "from_checkpoint", "load", "pack", "read_header",
           "run_metrics", "save", "to_checkpoint", "unpack", "verify"]
