import torch
import torch.nn as nn
import torch.nn.functional as F



class Head(nn.Module):
    
    """One head of self-attention"""

    def __init__(self,
                 n_emb,
                 head_size,
                 block_size,
                 dropout = None):

        super().__init__()
        self.n_emb = n_emb
        self.head_size = head_size
        self.block_size = block_size
        self.p = dropout

        self.key   = nn.Linear(n_emb, head_size, bias = False)
        self.query = nn.Linear(n_emb, head_size, bias = False)
        self.value = nn.Linear(n_emb, head_size, bias = False)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        '''
        Inputs:
            x    -- (B, T, C )
        Outputs:
            attn -- (B, T, hs)
        '''
        B,T,C = x.shape
        assert T <= self.block_size , f'Input sequence length {x.shape[1]=} exceeds max block size {self.block_size}'
        assert C == self.n_emb      , f'Input embedding dim {x.shape[2]=} does not match n_emb in attention layer {self.n_emb}'

        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        v = self.value(x) # (B, T, hs)

        scores = q @ k.transpose(-1,-2) / self.head_size**0.5                   # (B,T,hs) @ (B,hs,T) --> (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))      # (B,T,T)
        attn   = F.softmax(scores, dim = -1)
        if self.p is not None:
            attn   = self.dropout(attn)
        out   = attn @ v                                                       # (B,T,T) @ (B,T,hs)  --> (B,T,hs)

        return out
    



#-----------------------------------------------------


class MultiHeadAttention(nn.Module):

    """Multiple heads of self-attention"""

    def __init__(self,
                 n_emb,
                 n_heads,
                 head_size,
                 block_size,
                 dropout = None):

        super().__init__()
        self.p = dropout

        self.heads = nn.ModuleList([Head(n_emb, head_size, block_size, self.p) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_heads * head_size , n_emb)
        if self.p is not None:
            self.dropout = nn.Dropout(self.p)

    def forward(self, x):
        '''
        Inputs:
            x   -- (B, T, C)
        Outputs:
            out -- (B, T, C)
        '''

        out = torch.cat([h(x) for h in self.heads], dim = -1)       # (B,T, n_heads*hs)
        out = self.proj(out)                                        # (B,T, n_emb     )
        if self.p is not None:
            out = self.dropout(out)

        return out
    

#-----------------------------------------------------


class FeedForward(nn.Module):

    def __init__(self,
                n_emb,
                n_hidden = None,
                dropout  = None):

        super().__init__()

        self.n_emb    = n_emb
        self.n_hidden = n_hidden if n_hidden is not None else 4 * n_emb
        self.p        = dropout

        self.net = nn.Sequential(
            nn.Linear(n_emb, self.n_hidden),
            nn.GELU(),
            nn.Linear(self.n_hidden, n_emb)
        )

        if self.p is not None:
            self.net.append(nn.Dropout(self.p))

    def forward(self, x):
        return self.net(x)
    

#-----------------------------------------------------


class Block(nn.Module):

    """Transformer block"""

    def __init__(self,
                 n_emb,
                 n_heads,
                 block_size,
                 dropout = None):

        super().__init__()
        self.p         = dropout
        self.n_heads   = n_heads
        self.head_size = n_emb // n_heads

        self.sa = MultiHeadAttention(n_emb,
                                     n_heads,
                                     self.head_size,
                                     block_size,
                                     dropout=self.p)

        self.ff  = FeedForward(n_emb, dropout = self.p)

        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        '''
        Inputs:
            x -- (B, T, C)
        Outputs:
            x -- (B, T, C)
        '''
        x = x + self.sa( self.ln1(x) )
        x = x + self.ff( self.ln2(x) )

        return x