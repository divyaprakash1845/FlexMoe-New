import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_module import FMoETransformerMLP # MUST BE IN THE FOLDER

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv, attn_mask=None):
        Bx, Nx, Cx = x.shape
        B, N, C = kv.shape
        q = self.q(x).reshape(Bx, Nx, self.num_heads, Cx//self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(Bx, Nx, -1)
        x = self.proj(x)
        return self.proj_drop(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_experts, num_routers, d_model, num_head, dropout=0.1, mlp_sparse=False, top_k=2, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn = Attention(d_model, num_heads=num_head, attn_drop=dropout, proj_drop=dropout)
        self.mlp_sparse = mlp_sparse
        self.expert_index = None

        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=d_model, d_hidden=d_model * 2, activation=nn.GELU(), top_k=top_k, **kwargs)
        else:
            # Replaced the custom MLP with standard PyTorch layers to save space
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )

    def forward(self, x):
        # x is expected to be a list containing our tensor
        chunk_size = [item.shape[1] for item in x]
        x_cat = self.norm1(torch.cat(x, dim=1))
        
        attn_out = self.attn(x_cat, x_cat)
        x_cat = x_cat + self.dropout1(attn_out)
        
        x_split = list(torch.split(x_cat, chunk_size, dim=1))
        
        for i in range(len(chunk_size)):
            if self.mlp_sparse:
                x_split[i] = x_split[i] + self.dropout2(self.mlp(self.norm2(x_split[i]), self.expert_index))
            else:
                x_split[i] = x_split[i] + self.dropout2(self.mlp(self.norm2(x_split[i])))
        return x_split

class NeuroFlexMoE(nn.Module):
    def __init__(self, input_dim=9, seq_len=750, hidden_dim=64, num_layers=2, num_experts=4, num_routers=1, top_k=2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        layers = []
        _sparse = True
        for j in range(num_layers):
            layers.append(TransformerEncoderLayer(
                num_experts=num_experts, num_routers=num_routers, d_model=hidden_dim, num_head=4, mlp_sparse=_sparse, top_k=top_k
            ))
            _sparse = not _sparse
            
        self.network = nn.ModuleList(layers)
        self.decoder = nn.Linear(hidden_dim, 1) # REGRESSION HEAD FOR SUDS

    def forward(self, x):
        x = self.input_fc(x) + self.pos_embed[:, :x.size(1), :]
        x = [x] # Wrap in list to match FlexMoE chunk logic
        
        for layer in self.network:
            x = layer(x)
            
        x_out = x[0] # Unwrap from list
        out = self.decoder(x_out.mean(dim=1))
        return out.squeeze(-1)

    def gate_loss(self):
        # Extract the routing loss from the moe_module to balance the experts
        g_loss = []
        for mn, mm in self.named_modules():
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is not None:
                        g_loss.append(i_loss)
        if not g_loss:
            return torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        return sum(g_loss)
