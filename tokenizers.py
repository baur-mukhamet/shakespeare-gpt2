#Byte-Pair-Encoding algorithm

import torch
from collections import defaultdict
import re
from tqdm import tqdm





class BPE:
    """
    Byte-Pair-Encoding algorithm
    """
    def __init__(self, text):
        # characters in the text 
        self.chrs = sorted(list(set(text)))
        
        self.special_chrs = []
        self.alphabet = []

        for c in self.chrs:
            if c.isalpha():
                self.alphabet.append(c)
            else:
                self.special_chrs.append(c)

        
        # Init vocabulary to contain special (non-alphabetic) characters, unknown, bos, eos
        self.vocab = {c : i for i,c in enumerate(self.chrs)}
        self.vocab['<UNK>'] = len(self.vocab)
        self.vocab['<pad>'] = len(self.vocab)
        self.vocab['<bos>'] = len(self.vocab)
        self.vocab['<eos>'] = len(self.vocab)

        # Index to token for decoding
        self.itoc = {i : token for token, i in self.vocab.items() }
        
        self.vocab_size = len(self.vocab)
        self.max_token_len = max(len(k) for k in self.vocab)

        # compute stats
        self.raw_word_freq = defaultdict(int)
        self.token_freq = defaultdict(int)
        self.compute_token_freq(text)
    
    def compute_token_freq(self, text):
        words = sorted(re.findall(r"[a-z]+", text))
        for word in words:
            chrs_list = list(word)
            self.raw_word_freq[word] += 1
            self.token_freq['<bos> ' + ' '.join(chrs_list) + ' <eos>'] += 1

    def get_max_freq_pair(self, token_freq):
        pairs = defaultdict(int)
        for word, freq in token_freq.items():
            chrs = word.split()
            for i in range(len(chrs) - 1):
                pairs[(chrs[i], chrs[i+1])] += freq
        return max(pairs, key = pairs.get)

    # Merge max_freq_pair in token_freq
    def merge_symbols(self, max_freq_pair, token_freq):
        new_token_freq = {}
        for word, freq in token_freq.items():
            new_word = word.replace(' '.join(max_freq_pair), ''.join(max_freq_pair))
            new_token_freq[new_word] = freq
        return new_token_freq
    
    def build_vocab(self, vocab_size):
        num_merges = vocab_size - self.vocab_size

        for i in tqdm(range(num_merges)):
            max_freq_pair = self.get_max_freq_pair(self.token_freq)
            new_token = ''.join(max_freq_pair) 
            self.vocab[new_token] = self.vocab_size
            self.itoc[self.vocab_size] = new_token
            self.token_freq = self.merge_symbols(max_freq_pair, self.token_freq)
            self.vocab_size += 1
            self.max_token_len = max(self.max_token_len, len(new_token))



#-----------------------------------------------------


class Tokenizer:
    """Tokenizer based on BPE"""

    def __init__(self, bpe):
        self.vocab = bpe.vocab
        self.itoc = bpe.itoc
        self.alphabet = bpe.alphabet
        self.special_chrs = bpe.special_chrs
        self.max_token_len = bpe.max_token_len
        self.unk_id = self.vocab["<UNK>"]
        self.pad_token_id = self.vocab["<pad>"]
    

    def tokenize_to_subwords(self, word_seq):
        out = []

        for word in word_seq:
            if not word:
                continue
            elif word in self.special_chrs:
                out.append(word)
            else:
                word = "<bos>" + word + "<eos>"
                curr_out = []
                start = 0
                end = len(word)
                
                while start < end and start < len(word):
                    if word[start:end] in self.vocab:
                        curr_out.append(word[start:end])
                        start = end
                        end = len(word)
                    else:
                        end -= 1
                    
                if start < len(word):
                    curr_out.append('<UNK>')
                
                out.append(curr_out)

        return out
    

    def _tokenize(self, word_seq, block_len = None):
        # Tokenize into token ids 
        out = []

        for word in word_seq:
            if not word:
                continue
            elif word in self.special_chrs:
                out.append(self.vocab[word])
            else:
                word = "<bos>" + word + "<eos>" 
                start = 0
                end = len(word)
                
                while start < end and start < len(word):
                    curr_subword = word[start:end]
                    if curr_subword in self.vocab:
                        i = self.vocab[curr_subword]
                        out.append(i)
                        start = end
                        end = len(word)
                    else:
                        end -= 1
                    
                if start < len(word):
                    out.append(self.unk_id)
        

        out = torch.tensor(out, dtype = torch.long, requires_grad = False)
        
        if block_len is not None:
            # Split into block_len rows
            r = len(out) % block_len
            if r !=0:
                pad_tensor = self.pad_token_id * torch.ones(block_len - r, dtype = torch.long, requires_grad = False)
                out = torch.cat((out, pad_tensor))
            
            out = out.reshape((-1, block_len))
                            
        return out
    

    def tokenize(self, text, block_len = None):
        """
        In: 
            text - str 
        Out:
            out - list of int; tokenized text
        """

        pattern = re.compile(r'(\W|\s|\w+)')
        tokens = pattern.findall(text)

        return self._tokenize(tokens, block_len)
    

    def decode(self, token_ids):
        """
        Input:
            token_ids - torch tensor or list of int
        Output:
            sentence - str
        """

        sentence = ""

        if isinstance(token_ids, torch.Tensor) and token_ids.dim() == 2:
            token_ids = token_ids.flatten()
        
        for idx in token_ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            if idx == self.pad_token_id:
                continue
            
            subword = self.itoc[idx]

            bos = (subword[0:5] == "<bos>")
            eos = (subword[-5:] == "<eos>")

            start = len("<bos>") if bos else 0
            end = -len("<eos>") if eos else len(subword)
            
            sentence += subword[start:end]
            
        return sentence
    


#-----------------------------------------------------


class TokenizerChrs():
    """Simple character level tokenizer"""

    def __init__(self, chrs):
        self.stoi = {s:i for i,s in enumerate(chrs)}
        self.itos = {i:s for s,i in self.stoi.items()}


    def tokenize(self, chrs, block_size = None):
        
        out = torch.tensor([self.stoi[s] for s in chrs], dtype = torch.long, requires_grad = False)
        
        if block_size is not None:
            assert len(chrs) % block_size == 0, f'String length must be divisible by {block_size = }'
            out = out.view(-1, block_size)

        return out 


    def decode(self, idx):
        return ''.join([self.itos[i.item()] for i in idx])



# if __name__ == "__main__":
    
#     with open('input.txt', 'r') as f:
#         text = f.read().lower()
    
#     bpe = BPE(text)
#     bpe.build_vocab(1000)

#     assert bpe.vocab_size == 1000, "Vocab size incorrect"

#     tokenizer = Tokenizer(bpe.vocab)

#     out = tokenizer.tokenize(['first', 'citizen', 'here', 'lord', 'king', 'taller', 'table'])
