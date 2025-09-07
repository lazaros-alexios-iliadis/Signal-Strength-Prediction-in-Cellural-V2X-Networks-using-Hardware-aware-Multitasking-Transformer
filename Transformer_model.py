import torch.nn as nn
import torch


# Feature Tokenizer Module
class FeatureTokenizer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        if hasattr(x, 'dequantize'):
            x = x.dequantize()

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"[Tokenizer Error] Expected Tensor but got {type(x)}")

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.size(-1) != self.input_dim:
            raise ValueError(f"[Tokenizer Error] Expected {self.input_dim} features, got {x.size(-1)}")

        # Robust device move
        device = x.device if isinstance(x, torch.Tensor) else torch.device('cpu')
        if hasattr(self.embedding, "weight") and isinstance(self.embedding.weight, torch.Tensor):
            device = self.embedding.weight.device

        x = x.to(device)

        token = self.embedding(x).unsqueeze(1)    # [batch_size, 1, hidden_dim]
        token = token.expand(-1, self.input_dim, -1)  # [batch_size, input_dim, hidden_dim]

        return token


class MultiTaskFTTransformer(nn.Module):
    def __init__(self, input_dims, num_heads, num_layers, hidden_dim, output_dims):
        super().__init__()
        self.num_tasks = len(input_dims)

        self.feature_tokenizers = nn.ModuleList([
            FeatureTokenizer(dim, hidden_dim) for dim in input_dims
        ])

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, out_dim) for out_dim in output_dims
        ])

    def forward(self, x_list):
        if not isinstance(x_list, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(x_list)}")
        if len(x_list) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, but got {len(x_list)}")

        outputs = []
        device = next(self.parameters()).device

        for i, x in enumerate(x_list):
            if hasattr(x, 'dequantize'):
                x = x.dequantize()

            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

            x = x.to(device)

            # Tokenize and encode
            x_tok = self.feature_tokenizers[i](x)
            x_enc = self.transformer_encoder(x_tok)

            # Pool and output
            x_pool = x_enc.mean(dim=1)
            out = self.output_heads[i](x_pool)

            outputs.append(out)

        return outputs
