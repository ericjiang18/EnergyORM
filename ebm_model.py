# model.py
import torch
import torch.nn as nn

class TransEBM(nn.Module):
    """
    Lightweight Transformer-based Energy-Based Model.

    Args
    ----
    vocab_size : int
    d_model    : hidden / embedding size
    n_heads    : number of self-attention heads
    n_layers   : Transformer encoder depth
    dropout    : dropout rate for transformer layers
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        print(f"Initialized TransEBM: vocab={vocab_size}, d_model={d_model}, heads={n_heads}, layers={n_layers}, dropout={dropout}")

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        ids  : (B, L)  token ids
        mask : (B, L)  1 = real token, 0 = pad
        Returns
        -------
        energies : (B,)  lower = better
        """
        x = self.emb(ids) * (self.d_model**0.5)
        padding_mask = (mask == 0)
        x = self.enc(x, src_key_padding_mask=padding_mask)
        cls_representation = x[:, 0]
        energy = self.head(cls_representation).squeeze(-1)
        return energy

    def resize_token_embeddings(self, new_num_tokens: int):
        old_embeddings = self.emb
        if old_embeddings.num_embeddings == new_num_tokens:
            print(f"Token embedding size already matches ({new_num_tokens}). No resize needed.")
            return

        new_embeddings = nn.Embedding(new_num_tokens, self.d_model)
        new_embeddings.weight.data.normal_(mean=0.0, std=0.02) # Initialize new embeddings

        num_embeddings_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_embeddings_to_copy, :] = old_embeddings.weight.data[:num_embeddings_to_copy, :]

        self.emb = new_embeddings
        print(f"Resized token embeddings from {old_embeddings.num_embeddings} to {new_num_tokens}")