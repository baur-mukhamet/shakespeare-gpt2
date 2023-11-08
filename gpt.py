import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Block
import utils
from tqdm.notebook import tqdm


device = utils.device 


class GPT(nn.Module):

    def __init__(self,
                 vocab_size,
                 n_blocks,
                 n_emb,
                 n_heads,
                 block_size,
                 dropout = None):

        super().__init__()

        
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.n_emb = n_emb
        self.n_heads = n_heads
        self.block_size = block_size
        self.p = dropout

        self.token_emb = nn.Embedding(vocab_size, n_emb)
        self.pos_emb   = nn.Embedding(block_size, n_emb)

        self.blocks    = nn.Sequential(*[Block(n_emb, n_heads, block_size, self.p) for _ in range(n_blocks)])
        self.ln        = nn.LayerNorm(n_emb) # final layer norm
        self.lm_head   = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets = None):
        '''
        Inputs:
            idx     -- (B, T) vocab indices of input tokens
            targets -- (B, T) vocab indices of targets
        '''
        B,T = idx.shape

        emb = self.token_emb(idx)     # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=device))       # (T,C)

        x      = self.blocks(emb + pos)
        logits = self.lm_head( self.ln(x) )  # (B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy( logits.reshape(-1, self.vocab_size),
                                    targets.reshape(-1) )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        '''
        Inputs:
            idx -- (B,T) context indices
        '''
        for _ in tqdm(range(max_new_tokens)):
            idx_cropped = idx[:, -self.block_size : ]
            logits, _ = self(idx_cropped)                                   # (B,T,vocab_size)
            probs = F.softmax(logits, dim = -1)
            probs = probs[:,-1,:]
            idx_next = torch.multinomial(probs, 1)                 # (B, 1)
            idx = torch.cat([idx, idx_next], dim = -1)              # (B,T+1)

        return idx