import torch
import torch.nn as nn

from x_transformers.x_transformers import (
    exists,
    always,
    Encoder,
    AttentionLayers,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
)


def amp_scaler_factory(
    enabled: bool = True,
    init_scale: int = 65536,
    growth_factor: int = 2,
    growth_interval: int = 2000,
    backoff_factor: float = 0.5,
    verbose: bool = True
) -> torch.cuda.amp.GradScaler:
    scaler = torch.cuda.amp.GradScaler(
        enabled=enabled,
        init_scale=init_scale,
        growth_factor=growth_factor,
        growth_interval=growth_interval,
        backoff_factor=backoff_factor
    )
    if verbose: print(scaler)
    return scaler


class MyTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers,
        dim_in = None,
        dim_out = None,
        emb_dim = None,
        post_emb_norm = False,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), \
            "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.post_emb_norm = nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_in = nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()

        self.attn_layers = attn_layers

        self.project_out = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()

    def forward(
        self,
        x,
        return_embeddings = False,
        return_intermediates = False,
        mask = None,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        **kwargs
    ):

        x = self.project_in(x)
        x = x + self.pos_emb(x, pos = pos)

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            _, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(
            x, mask = mask, mems = mems,
            return_hiddens = True, **kwargs
        )

        out = self.project_out(x) if not return_embeddings else x

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out


def get_train_model(
    dim_in: int = 41,
    dim_out: int = 2,
    max_seq_len: int = 10,
    emb_dropout: float = 0.0,
    attention_dim: int = 16,
    attention_depth: int = 8,
    attention_head: int = 3,
    rel_pos_bias: bool = True,
    scaled_sinu_pos_emb: bool = False,
    use_abs_pos_emb: bool = True,
    post_emb_norm: bool = False,
    lr: float = 0.01,
    weight_decay: float = 1e-6,
    max_epochs: int = 1000,
    optimizer_name: str = "adamw",
    amp: bool = True,
    device: str = "cuda:0"
) -> torch.nn.Module:
    attn_layers = Encoder(
        dim=attention_dim,
        depth=attention_depth,
        heads=attention_head,
        rel_pos_bias=rel_pos_bias
    )
    model = MyTransformerWrapper(
        dim_in=dim_in,
        dim_out=dim_out,
        max_seq_len=max_seq_len,
        attn_layers=attn_layers,
        scaled_sinu_pos_emb=scaled_sinu_pos_emb,
        use_abs_pos_emb=use_abs_pos_emb,
        post_emb_norm=post_emb_norm,
        emb_dropout=emb_dropout,
    ).to(device)

    params = list(model.parameters())
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    else: assert 0

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #optimizer, T_max=max_epochs, eta_min=weight_decay, last_epoch=- 1, verbose=False
        optimizer, T_max=max_epochs, eta_min=0, last_epoch=- 1, verbose=False
    )

    if amp:
        scaler = amp_scaler_factory()
    else:
        scaler = None

    return model, optimizer, lr_scheduler, scaler
