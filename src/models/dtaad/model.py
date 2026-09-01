import torch

from ..shared.base import BaseDetector, DetectorOutput
from ._adapter_upstream import DTAADQASForecastAdapter


class DTAAD(BaseDetector):
    """DTAAD (Yu et al.) fitted to this repository's contract.

    Upstream, DTAAD reconstructs the last value already present in its input
    window using local and global TCN branches feeding a Transformer. Here the
    task is to forecast beyond the window: the local/global bodies are shared
    across horizons and only a small linear head per horizon is separate.

    Linear (identity) outputs replace the original sigmoid because the targets
    are standardized rather than confined to ``[0, 1]``. The two-phase output is
    preserved and exposed through ``auxiliary['phase1']``.
    """

    def __init__(
        self,
        context_length: int,
        horizon_steps: list,
        input_dim: int,
        output_dim: int,
        *args,
        **kwargs,
    ):
        super(DTAAD, self).__init__()

        if output_dim != input_dim:
            raise ValueError(
                "the DTAAD adapter reconstructs its own input channels: "
                f"input_dim={input_dim}, output_dim={output_dim}"
            )

        self.context_length = context_length
        self.horizon_steps = tuple(int(step) for step in horizon_steps)
        self.horizon_count = len(self.horizon_steps)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.init_modules()

    def init_modules(self):
        self.adapter = DTAADQASForecastAdapter(
            input_dim=self.input_dim,
            context_length=self.context_length,
            horizon_steps=self.horizon_steps,
        )

    def forward(self, context: torch.Tensor) -> DetectorOutput:
        self.check_context(context, self.context_length, self.input_dim)
        forecast = self.adapter(context)  # B, K, C
        auxiliary = {} if forecast.phase1 is None else {"phase1": forecast.phase1}
        return DetectorOutput(prediction=forecast.prediction, auxiliary=auxiliary)

    def get_config(self):
        model_args = {}
        model_args["name"] = "DTAAD"
        model_args["context_length"] = self.context_length
        model_args["horizon_steps"] = list(self.horizon_steps)
        model_args["input_dim"] = self.input_dim
        model_args["output_dim"] = self.output_dim
        model_args["adaptation"] = dict(self.adapter.adaptation_metadata)
        model_args["parameters"] = self.parameter_count

        return model_args
