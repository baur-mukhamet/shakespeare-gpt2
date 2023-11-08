#import torch
#import torch.nn as nn
#import torch.nn.functional as F
from gpt import GPT
import tokenizers
import trainer
import utils
from tqdm.notebook import tqdm
import argparse 
import os
import json
import itertools
#import sys



if __name__ == '__main__':

    device = utils.device # 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #----------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data', type = str, default = 'input.txt', help = 'Data file name')
    parser.add_argument('tokenizer', type = str, help = "Choose 'chrs' or 'bpe' tokenizer")
    parser.add_argument('--batch_sizes', nargs = '*', type = int, default = [64])
    parser.add_argument('--block_sizes', nargs = '*', type = int, default = [256])

    # model hyperparams
    parser.add_argument('--n_emb', type = int, default = 384, help = "Embedding dimension")
    parser.add_argument('--n_heads', type = int, default = 6, help = "Number of transformer heads")
    parser.add_argument('--n_blocks', type = int, default = 6, help = "Number of transformer layers")
    parser.add_argument('--dropouts', nargs = '*', type = float, default = [0.2])

    # training hyperparams
    parser.add_argument('--num_epochs', type = int, default = 10, help = "Number of training step")
    parser.add_argument('--eval_steps', type = int, default = 1000, help = "Periodicity of test/eval loss logging")
    parser.add_argument('--learning_rates', nargs = '*', type = float, default = [1e-3])
    parser.add_argument('--weight_decays', nargs = '*', type = float, default = [1e-2])

    # logging
    parser.add_argument('--save_losses', action = 'store_true')
    parser.add_argument('--save_checkpoint', action = 'store_true')
    parser.add_argument('--save_checkpoint_steps', type = int, default = 1000, help = 'Save after every save_checkpoint_steps steps')
    parser.add_argument('--run_dir', type = str, default = None) 

    # train from checkpoint
    parser.add_argument('--from_checkpoint', action = 'store_true')
    parser.add_argument('--from_checkpoint_run_dir', type = str, default = None)
    parser.add_argument('--from_checkpoint_step', type = int, default = None)

    # lr schedule
    parser.add_argument('--lr_schedule', type = str, default = 'const')
    parser.add_argument('--cosine_final_lr', nargs = '*', type = float, default = [1e-6])
    parser.add_argument('--cosine_T_max', nargs = '*', type = int, default = [10])
    parser.add_argument('--cosine_T_mult', nargs = '*', type = int, default = [1])
    parser.add_argument('--cosine_lr_restart_decay', nargs = '*', type = float, default = [1])

    args = parser.parse_args()

    #----------------------------------------------------------------------------------------

    with open(args.data, 'r') as f:
        text  = f.read()

    # data
    batch_sizes = args.batch_sizes
    block_sizes = args.block_sizes
    # model
    n_emb = args.n_emb
    n_heads = args.n_heads
    n_blocks = args.n_blocks
    dropouts = args.dropouts
    # training
    num_epochs = args.num_epochs
    eval_steps = args.eval_steps
    learning_rates = args.learning_rates
    weight_decays = args.weight_decays
    # logging
    save_losses = args.save_losses
    save_checkpoint = args.save_checkpoint
    save_checkpoint_steps = args.save_checkpoint_steps
    run_dir = args.run_dir

    # train from checkpoint
    from_checkpoint = args.from_checkpoint
    from_checkpoint_run_dir = args.from_checkpoint_run_dir 
    from_checkpoint_step = args.from_checkpoint_step

    # lr schedules
    lr_schedule = args.lr_schedule
    cosine_final_lr = args.cosine_final_lr
    cosine_T_max = args.cosine_T_max
    cosine_T_mult = args.cosine_T_mult
    cosine_lr_restart_decay = args.cosine_lr_restart_decay


    # create directory for logging, if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    if args.tokenizer == 'chrs':
        vocab = sorted(list(set(text)))

        # save vocabulary
        vocab_path = os.path.join(run_dir, 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(vocab,f)

        vocab_size = len(vocab)
        tokenizer = tokenizers.TokenizerChrs(vocab)

    elif args.tokenizer == 'bpe':
        pass

    else:
        raise ValueError("Argument tokenizer must be 'chrs' or 'bpe'") 


    num_experiments = (
                        len(batch_sizes)*len(block_sizes)*len(learning_rates)
                       *len(weight_decays)*len(dropouts)*len(cosine_final_lr)
                       *len(cosine_T_max)*len(cosine_T_mult)*len(cosine_lr_restart_decay)
                    )
    is_sweep = num_experiments > 1


    for i, (block_size, batch_size, lr, weight_decay, dropout, final_lr, T_max, T_mult, lr_restart_decay) in enumerate(itertools.product(block_sizes, 
                                                                                              batch_sizes, 
                                                                                              learning_rates, 
                                                                                              weight_decays, 
                                                                                              dropouts,
                                                                                              cosine_final_lr,
                                                                                              cosine_T_max,
                                                                                              cosine_T_mult,
                                                                                              cosine_lr_restart_decay
                                                                                              )):

        print(f"""\nExperiment {i+1}/{num_experiments}: 
{block_size=:>3} 
{batch_size=:>3} 
{lr=:>5.0e} 
{weight_decay=:>5.0e}
{dropout=:>5.0e}
{final_lr=:>5.0e}
{T_max=:>3}
{T_mult=:>3}
{lr_restart_decay=:>5.0e}"""
)

        model = GPT(vocab_size,
                    n_blocks,
                    n_emb,
                    n_heads,
                    block_size,
                    dropout).to(device)
        

        curr_run_dir = os.path.join(run_dir, f'run_{i+1}') if is_sweep else run_dir


        if i==0 or block_size != prev_block_size:
            data_dict = utils.prepare_dataset(text, tokenizer, block_size)
            prev_block_size = block_size
        
        trainer_ = trainer.Trainer( data_dict['train'],
                                    data_dict['eval'], 
                                    num_epochs, 
                                    batch_size, 
                                    eval_steps, 
                                    lr,
                                    weight_decay,
                                    save_losses,
                                    save_checkpoint,
                                    save_checkpoint_steps,
                                    curr_run_dir)
        
        trainer_.train(model, 
                       from_checkpoint, 
                       from_checkpoint_run_dir,
                       from_checkpoint_step,
                       lr_schedule = lr_schedule,
                       cosine_final_lr = final_lr,
                       cosine_T_max = T_max,
                       cosine_T_mult = T_mult,
                       cosine_lr_restart_decay = lr_restart_decay)