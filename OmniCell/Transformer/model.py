import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import torch
import torch.nn as nn
from typing import Optional
import einops
from flash_attn import  flash_attn_func

class LMConfig:
    def __init__(self,
                 num_gene:int = 60607,
                 d_model:int = 512,
                 token_per_cell:int = 500,
                 topk:int = 5,
                 num_shared:int = 5,
                 num_routing:int = 10,
                 heads:int = 8,
                 dropout:float = 0.1,
                 dim:int = 64,
                 base:float = 10000.0,
                 eps:float = 1e-6,
                 multiple_of:int = 512,
                 num_layers:int = 10,
                 rotary:bool=True ,
                 hidden_dim=512*4,
                 use_flash:bool = True):
        super().__init__()
        self.num_gene = num_gene
        self.d_model = d_model
        self.token_per_cell = token_per_cell
        self.heads = heads
        self.dropout = dropout
        self.dim = dim
        self.base = base
        self.eps = eps
        self.multiple_of = multiple_of
        self.rotary = rotary
        self.num_layers = num_layers
        self.num_shared = num_shared
        self.num_routing = num_routing
        self.topk = topk
        self.use_flash = use_flash
        self.hidden_dim = hidden_dim
           
class MoE4Embedder(nn.Module):
    def __init__(self, config:LMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_shared = config.num_shared
        self.num_routing = config.num_routing
        self.topk = config.topk
        self.load_balance_loss = 0.0

        self.shared_experts = nn.ModuleList([nn.Linear(1, self.d_model, bias=False) for _ in range(self.num_shared)])
        self.routing_experts = nn.ModuleList([nn.Linear(1, self.d_model, bias=False) for _ in range(self.num_routing)])
        
         
        self.router = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_model, self.num_routing, bias=False)
        )

    def forward(self, 
                gene_embedded:torch.Tensor, 
                value:torch.Tensor)->torch.Tensor:
         
        shared_input = value.unsqueeze(-1)
        shared_output = sum(expert(shared_input) for expert in self.shared_experts)
        
         
        routing_logits = self.router(gene_embedded)   
        
        routing_weights = F.softmax(routing_logits, dim=-1)   
        topk_weights, topk_idx = torch.topk(routing_weights, self.topk, dim=-1)
        
         
        sparse_weights = torch.zeros_like(routing_weights).scatter(
            -1, topk_idx, topk_weights
        )   
        
         
         
        expert_outputs = torch.stack([
            expert(shared_input) for i, expert in enumerate(self.routing_experts)
        ], dim=2)   
        routing_output = (expert_outputs * sparse_weights.unsqueeze(-1)).sum(dim=2)
        
         
        self.load_balance_loss = self._calc_balance_loss(routing_weights, sparse_weights)
        return shared_output + routing_output

    def _calc_balance_loss(self, 
                           routing_weights:torch.Tensor, 
                           sparse_weights:torch.Tensor)->torch.Tensor:
         
        routing_mask = (sparse_weights > 0).float()
        
         
        N_prime = self.num_routing
        K_prime = self.topk
        T = sparse_weights.size(0) * sparse_weights.size(1)
        
         
        expert_count = routing_mask.sum([0,1])   
        f_i = (expert_count / (K_prime * T)) * N_prime   
        
         
        P_i = routing_weights.mean([0,1])   
        
         
        return (f_i * P_i).sum()

class Embedder(nn.Module):
    def __init__(self,config:LMConfig):
        super().__init__()
        self.load_balance_loss = 0.0
        self.gene_embedding = nn.Embedding(num_embeddings=config.num_gene, embedding_dim=config.d_model)
        
        self.value_embedding = MoE4Embedder(config)
        self.load_balance_loss = self.value_embedding.load_balance_loss
    def forward(self,
                gene:torch.Tensor,
                value:torch.Tensor) -> torch.Tensor:
        """

        Args:
            gene (torch.Tensor): [batch_size,tgt_len]
            value (torch.Tensor): [batch_size,tgt_len]
            cell (torch.Tensor): [batch_size,tgt_len]

        Returns:
            torch.Tensor: [batch_size,tgt_len,d_model]
        """
        gene_embedded = self.gene_embedding(gene)
        value = value.to(gene_embedded.dtype)
        value_embedded = self.value_embedding(gene_embedded,value)
        self.load_balance_loss = self.value_embedding.load_balance_loss
        fused = gene_embedded + value_embedded
         
        return fused

class RMSNorm(nn.Module):
    def __init__(self, config:LMConfig):
        super().__init__()
        self.eps = config.eps
        self.gamma = nn.Parameter(torch.ones(config.d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
         
        x_fp32 = x.float()
        return x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
         
        normalized = self._norm(x).type_as(x)
        return normalized * self.gamma

class FeedForward(nn.Module):
    def __init__(self, config:LMConfig):
        super().__init__()
        
         
        if config.hidden_dim is None:
            expanded_dim = 4 * config.d_model
            hidden_dim = int(2 * expanded_dim / 3)
            hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of -1) // config.multiple_of)
        else:
            hidden_dim = config.hidden_dim   
        
         
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim,config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_gate = F.silu(self.gate_proj(x))   
        up = self.up_proj(x)                     
        return self.dropout(self.down_proj(swish_gate * up))

class SelfAttention(nn.Module):
    def __init__(self, config:LMConfig):
        super().__init__()
        assert config.d_model % config.heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = config.d_model
        self.heads = config.heads
        self.head_dim = config.d_model // config.heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = getattr(config, 'use_flash', True)   
        self.rotary = getattr(config, 'rotary', False)

         
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

         
        if self.rotary:
            self.rotary_embedding = RotaryPositionalEncoding(config)

         
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.1))

    def forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入:
            x: [batch_size, seq_len, d_model]
            index: [batch_size, seq_len, 2]
        输出:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)   
        k = self.k_proj(x)
        v = self.v_proj(x)

         
        q = q.view(batch_size, seq_len, self.heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.heads, self.head_dim)

         
        if self.rotary and index is not None:
            q = self.rotary_embedding(q, index)
            k = self.rotary_embedding(k, index)

         
        if self.use_flash:
             
            attn_output = self._flash_attention(q, k, v)
        else:
             
            attn_output = self._vanilla_attention(q, k, v)

         
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)   

         
        return self.out_proj(attn_output)

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """优化的Flash Attention实现"""
        
         
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        
        return flash_attn_func(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,   
            return_attn_probs=False
        )

    def _vanilla_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        
        Q = Q * self.scale
        
         
        return F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,   
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False   
        )

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        assert config.dim % 4 == 0 
        
        self.dim = config.dim
        self.base = config.base
        
        self.half_dim = config.dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_rotation_matrix(self, index: torch.Tensor) -> torch.Tensor:
        B, T, _ = index.shape
        expanded_index = index.repeat(1, 1, self.half_dim // 2)
        inv_freq_broadcasted = self.inv_freq.view(1, 1, -1)
        freqs = expanded_index.float() * inv_freq_broadcasted
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        B, T, H, Dh = x.shape   

         
        x_rot = x[..., :self.dim]
        x_pass = x[..., self.dim:]

         
        x_rot = x_rot.view(B, T, H, -1, 2)
        x_rot = x_rot.float()
        x_complex = torch.view_as_complex(x_rot.contiguous())

         
        pos_cis = self._compute_rotation_matrix(index)
        pos_cis = pos_cis.unsqueeze(2)   

         
        x_rotated = torch.view_as_real(x_complex * pos_cis)
        x_rotated = x_rotated.view(B, T, H, -1)

        x_rotated = x_rotated.to(x_pass.dtype)
        return torch.cat([x_rotated, x_pass], dim=-1)
        
class TransformerBlock(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.norm_attn = RMSNorm(config)        
        self.self_attn = SelfAttention(config)  
        self.norm_ffn = RMSNorm(config)         
        self.ffn = FeedForward(config)          
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
         
        attn_output = self.self_attn(self.norm_attn(x), index)
        x = x + self.dropout(attn_output)
        
         
        ffn_output = self.ffn(self.norm_ffn(x))
        x = x + self.dropout(ffn_output)
        return x

class Transformer(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.embedder = Embedder(config)

         
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
    
        self.output = Output(config=config)
         
        self.load_balance_loss = 0.0
        self._init_weights()
        
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                 
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                 
                nn.init.ones_(module.gamma)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            
    def forward(self,
               gene: torch.Tensor,   
               value: torch.Tensor,   
               index: torch.Tensor) -> torch.Tensor:   

         
        x = self.embedder(gene, value)   
        self.load_balance_loss = self.embedder.load_balance_loss

         
        for layer in self.layers:
            x = layer(x, index)
        
        return x

class Output(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.token_per_cell = config.token_per_cell
        
         
        self.W = nn.Parameter(torch.Tensor(config.d_model, config.d_model))
        self.reset_parameters()
    
    def reset_parameters(self):
         
        nn.init.xavier_uniform_(self.W)
    def _extract_middle_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            token_per_cell: 每个细胞中需要保留的有效 token 数量
        Returns:
            [batch_size, num_cell * token_per_cell, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
         
        cell_length = self.token_per_cell + 2
        
         
        num_cell = seq_len // cell_length
        
         
        x_reshaped = x.view(batch_size, num_cell, cell_length, d_model)
        
         
        selected = x_reshaped[:, :, 1:-1, :]   
        
         
        return selected.reshape(batch_size, num_cell * self.token_per_cell, d_model)
    
    def forward(self, x):
        x = self._extract_middle_tokens(x)
        B, S, D = x.shape
        assert S % self.token_per_cell == 0, "序列长度必须能被token_per_cell整除"
        tpc = self.token_per_cell
        nc = S // tpc   

         
        x_blocks = x.view(B, nc, tpc, D)
        
         
        pooled = x_blocks.mean(dim=2)

         
        W_sym = (self.W + self.W.T) / 2

         
        transformed_pooled = torch.matmul(pooled, W_sym)

         
         
        transformed_expanded = transformed_pooled.unsqueeze(2)
         
        scores = torch.einsum('bnid,bnjd->bnj', transformed_expanded, x_blocks)

         
        return scores.reshape(B, S).unsqueeze(-1)


            