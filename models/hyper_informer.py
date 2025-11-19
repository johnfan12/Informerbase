from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import Informer, InformerStack


class SeriesEncoder(nn.Module):
    """Pool encoder outputs into a task-level representation."""

    def __init__(self, d_model: int, z_dim: int, pool: str = 'mean'):
        super().__init__()
        self.proj = nn.Linear(d_model, z_dim)
        self.pool = pool

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """enc_out: [B, L, d_model] -> z: [B, z_dim]."""
        if self.pool == 'last':
            pooled = enc_out[:, -1, :]
        else:
            pooled = enc_out.mean(dim=1)
        return self.proj(pooled)


class HyperHead(nn.Module):
    """Hyper-network that predicts a per-sample linear projection."""

    def __init__(self, z_dim: int, d_hid: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_w = nn.Linear(hidden_dim, d_hid * out_dim)
        self.fc_b = nn.Linear(hidden_dim, out_dim)
        self.d_hid = d_hid
        self.out_dim = out_dim

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return batch-specific weight and bias tensors."""
        hidden = F.relu(self.fc1(z))
        hidden = F.relu(self.fc2(hidden))
        W = self.fc_w(hidden).view(-1, self.d_hid, self.out_dim)
        b = self.fc_b(hidden)
        return W, b


class HyperInformer(nn.Module):
    """Wrap an Informer backbone with a hyper-network prediction head."""

    BACKBONES = {
        'informer': Informer,
        'informerstack': InformerStack,
    }

    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn='prob',
        embed='fixed',
        freq='h',
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True,
        device=torch.device('cuda:0'),
        backbone_type: str = 'informer',
        stack_layers=None,
        z_dim: int = 128,
        hyper_hidden_dim: int = 128,
        pool: str = 'mean',
    ):
        super().__init__()
        if backbone_type not in self.BACKBONES:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}")
        backbone_cls = self.BACKBONES[backbone_type]
        backbone_e_layers = stack_layers if backbone_type == 'informerstack' and stack_layers is not None else e_layers
        backbone_kwargs = dict(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=seq_len,
            label_len=label_len,
            out_len=out_len,
            factor=factor,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=backbone_e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn=attn,
            embed=embed,
            freq=freq,
            activation=activation,
            output_attention=output_attention,
            distil=distil,
            mix=mix,
            device=device,
        )
        self.backbone = backbone_cls(**backbone_kwargs)
        self.pred_len = out_len
        self.c_out = c_out
        self.d_model = d_model

        self.series_encoder = SeriesEncoder(d_model, z_dim, pool=pool)
        self.hyper_head = HyperHead(z_dim, d_model, out_len * c_out, hidden_dim=hyper_hidden_dim)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out, dec_out, _ = self.backbone.forward_features(
            x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=enc_self_mask,
            dec_self_mask=dec_self_mask,
            dec_enc_mask=dec_enc_mask,
        )
        z = self.series_encoder(enc_out)
        dec_pred = dec_out[:, -self.pred_len:, :]
        h = dec_pred.mean(dim=1)
        W, b = self.hyper_head(z)
        y = torch.bmm(h.unsqueeze(1), W).squeeze(1) + b
        return y.view(-1, self.pred_len, self.c_out)
