import torch

from ..shared import autoencoder, decoder, encoder
from ..shared.base import BaseDetector, DetectorOutput
from ..shared.layers import MultiHorizonPredictionHeads


class DualTrAD(BaseDetector):
    """Dual-evidence Transformer anomaly detector.

    One shared causal encoder feeds two branches: a multi-horizon forecaster and
    a denoising autoencoder. Each branch yields its own residual, and the two
    residuals are fused into a single anomaly score outside the model.

    "Dual" refers to the two *kinds of evidence*, not to two branches of the
    same kind and not to two encoders. Sharing one encoder is deliberate: both
    evidences come from the same representation, so a gain from combining them
    is attributable to the combination rather than to a second trained model.

    Setting ``autoencoder_params['enabled']`` to false leaves a forecasting-only
    detector on the same skeleton, which is the reference point used in the
    ablation.
    """

    def __init__(
        self,
        context_length: int,
        horizon_count: int,
        input_dim: int,
        output_dim: int,
        encoder_params: dict,
        decoder_params: dict,
        head_params: dict,
        autoencoder_params: dict = None,
        *args,
        **kwargs,
    ):
        super(DualTrAD, self).__init__()

        self.context_length = context_length
        self.horizon_count = horizon_count
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_params = dict(encoder_params)
        self.decoder_params = dict(decoder_params)
        self.head_params = dict(head_params)
        self.autoencoder_params = dict(autoencoder_params or {})

        self.init_modules()

    def init_modules(self):
        self.encoder: encoder.BaseEncoder = encoder.get(self.encoder_params["encoder_type"])(
            **self.encoder_params,
            in_chan=self.input_dim,
            max_length=self.context_length,
        )
        self.enc_out_chan = self.encoder.get_out_chan()

        self.horizon_decoder: decoder.BaseHorizonDecoder = decoder.get(
            self.decoder_params["decoder_type"]
        )(**self.decoder_params, horizon_count=self.horizon_count, d_model=self.enc_out_chan)

        self.prediction_heads = MultiHorizonPredictionHeads(
            **self.head_params,
            horizon_count=self.horizon_count,
            d_model=self.enc_out_chan,
            output_dim=self.output_dim,
        )

        self.autoencoder = None
        self.latent_alignment = None
        if self.autoencoder_params.get("enabled", False):
            self.autoencoder = autoencoder.DenoisingAutoencoderBranch(
                d_model=self.enc_out_chan,
                bottleneck_dim=self.autoencoder_params["bottleneck_dim"],
                decoder_hidden_dim=self.autoencoder_params["decoder_hidden_dim"],
                context_length=self.context_length,
                output_dim=self.output_dim,
                dropout=self.autoencoder_params.get("dropout", 0.1),
            )
            if self.autoencoder_params.get("align_latents", True):
                self.latent_alignment = autoencoder.LatentAlignment(
                    bottleneck_dim=self.autoencoder_params["bottleneck_dim"],
                    d_model=self.enc_out_chan,
                )

    def forward(self, context: torch.Tensor) -> DetectorOutput:
        self.check_context(context, self.context_length, self.input_dim)

        memory = self.encoder(context)  # B, L, F -> B, L, D

        prediction_features = self.horizon_decoder(memory)  # B, K, D
        prediction = self.prediction_heads(prediction_features)  # B, K, C

        reconstruction = None
        reconstruction_latent = None
        aligned_reconstruction = None
        if self.autoencoder is not None:
            reconstruction, reconstruction_latent = self.autoencoder(memory)  # B, L, C
            if self.latent_alignment is not None:
                aligned_reconstruction = self.latent_alignment(reconstruction_latent)  # B, D

        return DetectorOutput(
            prediction=prediction,
            reconstruction=reconstruction,
            prediction_features=prediction_features,
            reconstruction_latent=reconstruction_latent,
            aligned_reconstruction=aligned_reconstruction,
        )

    def get_config(self):
        model_args = {}
        model_args["name"] = "DualTrAD"
        model_args["context_length"] = self.context_length
        model_args["horizon_count"] = self.horizon_count
        model_args["input_dim"] = self.input_dim
        model_args["output_dim"] = self.output_dim
        model_args["encoder"] = self.encoder.get_config()
        model_args["horizon_decoder"] = self.horizon_decoder.get_config()
        model_args["prediction_heads"] = self.prediction_heads.get_config()
        model_args["autoencoder"] = (
            self.autoencoder.get_config() if self.autoencoder is not None else None
        )
        model_args["latent_alignment"] = (
            self.latent_alignment.get_config() if self.latent_alignment is not None else None
        )
        model_args["parameters"] = self.parameter_count

        return model_args
