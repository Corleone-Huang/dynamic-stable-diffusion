import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction

###################################################################################
# input: noised image x_t [B, C_v, H, W], timestep t [B], original text features context [B, L, C_w]
# output: refined text features 
###################################################################################

# base function
def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, visual_dim, textual_dim=None, heads=8, dropout=0.):
        super().__init__()
        dim_head = textual_dim // heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(textual_dim, textual_dim, bias=False)
        self.to_k = nn.Linear(visual_dim, textual_dim, bias=False)
        self.to_v = nn.Linear(visual_dim, textual_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(textual_dim, textual_dim),
            nn.Dropout(dropout)
        )

    def forward(self, visual, textual):
        h = self.heads

        q = self.to_q(textual)
        k = self.to_k(visual)
        v = self.to_v(visual)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out), attn

class SimplifiedCrossAttention(nn.Module):
    def __init__(self, visual_dim, textual_dim, heads=8, ):
        super().__init__()
        self.heads = heads
        self.proj = nn.Linear(visual_dim, textual_dim, bias=False)

    def forward(self, visual, textual):
        visual_context = self.proj(visual)

        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (textual, visual_context))
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)

        visual_contextual_features = einsum('b h i j, b h j d -> b h i d', attn, k)
        visual_contextual_features = rearrange(visual_contextual_features, "b h i d -> b i (h d)")

        return visual_contextual_features, attn

class SimpleAttentionSemanticEvolution(nn.Module):
    def __init__(self, visual_dim, textual_dim, heads, dropout, version="vanilla") -> None:
        super().__init__()
        if version == "vanilla":
            self.cross_attn = CrossAttention(
                visual_dim=visual_dim, textual_dim=textual_dim, heads=heads, dropout=dropout
            )
        elif version == "simple":
            self.cross_attn = SimplifiedCrossAttention(
                visual_dim, textual_dim, heads
            )
    

    def forward(self, x, timesteps, context, mask=None):
        # x: image features
        # context: text features
        x = rearrange(x, 'b c h w -> b (h w) c')

        refined_context = self.cross_attn(visual=x, textual=context)[0]

        return refined_context


if __name__ == "__main__":
    visual = torch.randn(10, 16, 16, 16)
    textual = torch.randn(10, 77, 768)

    model = SimpleAttentionSemanticEvolution(visual_dim=16, textual_dim=768, heads=12, dropout=0.1, version="vanilla")
    model(x=visual, timesteps=None, context=textual)