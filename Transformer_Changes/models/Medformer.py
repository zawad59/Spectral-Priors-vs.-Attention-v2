# Medformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer
from layers.Embed import ListPatchEmbedding


class Model(nn.Module):
    """
    Medformer-style encoder model.

    Notes on attention extraction:
      - This model will only return attention maps if `configs.output_attention == True`
        at *construction time* (because that flag is passed into MedformerLayer()).
      - So set: `configs.output_attention = True` BEFORE you do `model = Model(configs)`.
    """

    def __init__(self, configs):
        super().__init__()

        # -------- basic config --------
        self.task_name = configs.task_name
        self.pred_len = getattr(configs, "pred_len", None)

        # IMPORTANT: do NOT set configs.output_attention here.
        # Set it BEFORE instantiation: configs.output_attention = True
        self.output_attention = bool(getattr(configs, "output_attention", False))

        self.enc_in = configs.enc_in
        self.single_channel = bool(getattr(configs, "single_channel", False))

        # -------- embedding / patching --------
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list  # your original choice
        seq_len = configs.seq_len

        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        augmentations = configs.augmentations.split(",")

        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in,
            configs.d_model,
            patch_len_list,
            stride_list,
            configs.dropout,
            augmentations,
            configs.single_channel,
        )

        # -------- encoder --------
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        # IMPORTANT: this must be True at init-time to propagate attn
                        bool(getattr(configs, "output_attention", False)),
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # -------- heads --------
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)

            in_dim = (
                configs.d_model
                * sum(patch_num_list)
                * (1 if not self.single_channel else configs.enc_in)
            )

            self.projection = nn.Linear(in_dim, configs.num_class)

        else:
            # Keep placeholders for other tasks (as in your original)
            pass

    # -----------------------------
    # Task-specific methods
    # -----------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc=None):
        """
        Returns:
          - logits (B, num_class)
          - if output_attention=True: (logits, attns)
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # If using per-channel mode, reshape as your original code did
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        # Classification head
        output = self.act(enc_out)
        output = self.dropout(output)

        # Flatten: (batch, seq*d_model*...)
        output = output.reshape(output.shape[0], -1)

        logits = self.projection(output)

        if self.output_attention:
            return logits, attns
        return logits

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

        if self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)

        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)

        return None