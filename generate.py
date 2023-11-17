import torch
from gpt import GPT
import argparse
import json
import os
import utils 
import tokenizers
import bpe


if __name__ == '__main__':
    device = utils.device

    #----------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('run_dir', type = str, help = 'Run directory with model/training hyperparams, state dictionary and losses')
    parser.add_argument('--tokenizer_type', type = str, default = 'chrs', help = 'Tokenizer is either "chrs" or "bpe"' )
    parser.add_argument('--checkpoint_step', nargs = '*', type = int, help = "Checkpoint steps")
    
    parser.add_argument('--best_eval_loss_model', action = 'store_true', help = 'Load best eval loss model. Overrides checkpoint_step argument')
    parser.add_argument('--max_new_tokens', type = int, default = 100, help = 'Max new tokens to generate')
    parser.add_argument('--print', action = 'store_false')
    parser.add_argument('--write_to_file', type = str, default = None, help = "File path")

    args = parser.parse_args()
    #----------------------------------------------------------------------------------------

    vocab_path = os.path.join(args.run_dir, 'vocab.json')

    if args.best_eval_loss_model:

        best_eval_loss, epoch, checkpoint_dir = utils.find_best_eval_loss_model_2(args.run_dir)

        print(f'Best eval loss model is:\n{checkpoint_dir=}\ncheckpoint_step={best_eval_loss[1]}\nbest_eval_loss={best_eval_loss[0]:.4f}\n\n')

        checkpoint_step =[epoch] # [best_eval_loss[1]] # to run a loop over steps

    else:
        checkpoint_dir = args.run_dir
        checkpoint_step = args.checkpoint_step


    for checkpoint in checkpoint_step:

        # define paths to model hyperparms and state dict in run_dir
        model_hyperparams_path = os.path.join(checkpoint_dir, 'model_hyperparameters.json')
        state_dict_path = os.path.join(checkpoint_dir, f'checkpoint_{checkpoint}.pth')
        
        

        # load parameters
        with open(model_hyperparams_path, 'r') as f:
            model_hyperparms = json.load(f)
        state_dict = torch.load(state_dict_path)
        
        # model
        model = GPT(**model_hyperparms).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        

        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        new_text_encoded = model.generate(context, max_new_tokens=args.max_new_tokens)[0] # (T,)
        
        if args.tokenizer_type == 'chrs':

            # generate
            try:
                with open(vocab_path, 'r') as f:
                    vocab = json.load(f)
            except FileNotFoundError:
                # search in the parent directory
                parent = os.path.dirname(args.run_dir)
                vocab_path = os.path.join(parent, 'vocab.json')
                with open(vocab_path, 'r') as f:
                    vocab = json.load(f)

            tokenizer = tokenizers.TokenizerChrs(vocab)    
        
        elif args.tokenizer_type == 'bpe':
            tokenizer = bpe.BPETokenizer()
            
        else:
            raise ValueError("Tokenizer type must be 'chrs' or 'bpe'")
        
        new_text = tokenizer.decode(new_text_encoded)

        if args.write_to_file is not None:
            new_text_path = os.path.join(args.run_dir, args.write_to_file)
            with open(new_text_path, 'a') as f:
                f.write(f'Checkpoint {checkpoint} output:\n\n')
                f.write(new_text + '\n\n')
        
        if args.print == True:
            if len(checkpoint_step) > 1: 
                print(f'Checkpoint {checkpoint} output:\n\n')

            print(new_text + '\n\n')

