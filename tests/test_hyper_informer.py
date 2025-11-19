import torch

from models.hyper_informer import HyperInformer


def test_hyper_informer_forward_shape():
    model = HyperInformer(
        enc_in=4,
        dec_in=4,
        c_out=2,
        seq_len=32,
        label_len=16,
        out_len=8,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        embed='timeF',
        backbone_type='informer',
        z_dim=16,
        hyper_hidden_dim=32,
    )
    batch = 2
    x_enc = torch.randn(batch, 32, 4)
    x_dec = torch.randn(batch, 24, 4)
    x_mark_enc = torch.zeros(batch, 32, 4)
    x_mark_dec = torch.zeros(batch, 24, 4)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    assert out.shape == (batch, 8, 2)
