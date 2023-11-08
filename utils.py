import torch
from tqdm.notebook import tqdm
import os
import json



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def set_seed(seed = 1991):
    torch.manual_seed(seed)




@torch.no_grad()
def estimate_loss(model, 
                  data_train, 
                  data_eval, 
                  batch_size, 
                  eval_iter):
    
    model.eval()
    
    out = {}
    for split, data in zip(['train', 'val'], [data_train, data_eval]):
        losses = torch.zeros(eval_iter)
        for i in tqdm(range(eval_iter), leave = False):
            batch = get_batch(data, batch_size)
            logits, loss = model(batch[0], batch[1])
            losses[i] = loss.item()
        out[split] = losses.mean()

    model.train()

    return out


def prepare_dataset(text, tokenizer, block_size):

    text_cropped = text[:len(text)-len(text)%block_size + 1] # Crop to lenght (n*block_size + 1)
    X = tokenizer.tokenize(text_cropped[:-1], block_size)    # (B,T)
    Y = tokenizer.tokenize(text_cropped[1:], block_size)     # (B,T)
    data = torch.stack((X,Y))                                # (2,B,T)

    data = data.transpose(0,1).contiguous()   # (2,B,T) --> (B,2,T) ; DataLoader expects batch dim first

    # train/val/test split
    n1 = int(0.8 * data.shape[0])
    n2 = int(0.9 * data.shape[0])
    data_dict = {'train' : data[:n1],
                    'eval'   : data[n1:n2],
                    'test'  : data[n2:]}
    
    return data_dict


def find_best_eval_loss_model(run_dir, best_eval_loss = (float('inf'),-1), checkpoint_dir = ''):
    
    losses_path = os.path.join(run_dir, 'losses.json')
    
    if os.path.isfile(losses_path):
        with open(losses_path, 'r') as f:
            losses = json.load(f)
        
        if losses['best_eval_loss'][0] < best_eval_loss[0]:
            best_eval_loss = losses['best_eval_loss']
            checkpoint_dir = run_dir 

    for filename in os.listdir(run_dir):
        sub_dir = os.path.join(run_dir, filename)
        if os.path.isdir(sub_dir):
            best_eval_loss, checkpoint_dir = find_best_eval_loss_model(sub_dir, best_eval_loss, checkpoint_dir)
    
    return best_eval_loss, checkpoint_dir



def find_best_eval_loss_model_2(run_dir, best_eval_loss = (float('inf'),-1), checkpoint_dir = ''):
    
    losses_path = os.path.join(run_dir, 'losses.json')
    
    if os.path.isfile(losses_path):
        with open(losses_path, 'r') as f:
            losses = json.load(f)
        
        if losses['best_eval_loss'][0] < best_eval_loss[0]:
            best_eval_loss = losses['best_eval_loss']
            epoch = losses['steps'].index(best_eval_loss[1]) + 1
            checkpoint_dir = run_dir 

    for filename in os.listdir(run_dir):
        sub_dir = os.path.join(run_dir, filename)
        if os.path.isdir(sub_dir):
            best_eval_loss, epoch, checkpoint_dir = find_best_eval_loss_model(sub_dir, best_eval_loss, checkpoint_dir)
    
    return best_eval_loss, epoch, checkpoint_dir


# def find_worst_eval_loss_model(run_dir, best_eval_loss = (0.,-1), checkpoint_dir = ''):
    
#     losses_path = os.path.join(run_dir, 'losses.json')
    
#     if os.path.isfile(losses_path):
#         with open(losses_path, 'r') as f:
#             losses = json.load(f)
        
#         if losses['best_eval_loss'][0] > best_eval_loss[0]:
#             best_eval_loss = losses['best_eval_loss']
#             checkpoint_dir = run_dir 

#     for filename in os.listdir(run_dir):
#         sub_dir = os.path.join(run_dir, filename)
#         if os.path.isdir(sub_dir):
#             best_eval_loss, checkpoint_dir = find_worst_eval_loss_model(sub_dir, best_eval_loss, checkpoint_dir)
    
#     return best_eval_loss, checkpoint_dir





#----------Sampling batches from datasets------------------------------------------------------


# Simple sampling with replacement
def get_batch(data, batch_size):
    idx = torch.randint(data.shape[1], (batch_size,))
    out = data[:, idx, :]
    out = out.to(device)

    return out


# Sampling without replacement. Less noisy, each example sampled only once per epoch.
class DataLoader:

    def __init__(self, data, batch_size, shuffle = True):
        """
        data -- torch.tensor with batch dim first (B,...)
        """

        self.data = data
        self.batch_size = batch_size
        self.B = data.shape[0]
        self.shuffle = shuffle

    def __iter__(self):
        # Iterator state 
        self.curr_iter = 0
        # Optionally shuffle
        self.perm_data = self.data[torch.randperm(self.B)] if self.shuffle else self.data

        return self

    def __next__(self):
            
        if self.curr_iter >= len(self):
            raise StopIteration

        batch = self.perm_data[self.curr_iter * self.batch_size : (self.curr_iter + 1) * self.batch_size]
        self.curr_iter += 1

        batch = batch.to(device)

        return batch

    def __len__(self):
        l = self.B // self.batch_size
        if self.B % self.batch_size != 0:
            l += 1
        return l
