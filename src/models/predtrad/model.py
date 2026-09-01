import torch

from ..shared.base import BaseDetector, DetectorOutput
from ._adapter_upstream import PredTrADV1QASAdapter


class PredTrADv1(BaseDetector):
    """PredTrAD-V1 (Schuster et al.) fitted to this repository's contract.

    The upstream network predicts the next point of a shifted sequence. To reach
    the horizons used here it is rolled out autoregressively: each step appends
    the model's own last prediction and re-runs, so no future target ever enters
    the input. Only the steps that match a requested horizon are kept.

    The upstream body is vendored verbatim under ``models/third_party`` with a
    SHA-256 record; this class only wires it to the shared interface.
    """

    def __init__(
        self,
        context_length: int,
        horizon_steps: list,
        input_dim: int,
        output_dim: int,
        core_params: dict = None,
        *args,
        **kwargs,
    ):
        super(PredTrADv1, self).__init__()

        if output_dim != input_dim:
            raise ValueError(
                "PredTrAD-V1 rolls its own prediction back into the input, so it "
                f"must forecast every input channel: input_dim={input_dim}, "
                f"output_dim={output_dim}"
            )

        self.context_length = context_length
        self.horizon_steps = tuple(int(step) for step in horizon_steps)
        self.horizon_count = len(self.horizon_steps)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.core_params = dict(core_params or {})

        self.init_modules()

    def init_modules(self):
        self.adapter = PredTrADV1QASAdapter(
            **self.core_params,
            input_dim=self.input_dim,
            horizons=self.horizon_steps,
        )

    def forward(self, context: torch.Tensor) -> DetectorOutput:
        self.check_context(context, self.context_length, self.input_dim)
        forecast = self.adapter(context)  # B, K, C
        return DetectorOutput(prediction=forecast.prediction)

    def get_config(self):
        model_args = {}
        model_args["name"] = "PredTrAD_v1"
        model_args["context_length"] = self.context_length
        model_args["horizon_steps"] = list(self.horizon_steps)
        model_args["input_dim"] = self.input_dim
        model_args["output_dim"] = self.output_dim
        model_args["adaptation"] = dict(self.adapter.adaptation_metadata)
        model_args["parameters"] = self.parameter_count

        return model_args
