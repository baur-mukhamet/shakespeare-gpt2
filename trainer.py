import torch
import utils 
import os
import json
from tqdm.notebook import tqdm
from lr_scheduler import CosineScheduleLR, CosineScheduleWarmRestarts



class Trainer:

    def __init__(self,
                 data_train,
                 data_eval, 
                 num_epochs, 
                 batch_size, 
                 eval_steps,
                 learning_rate,
                 weight_decay,
                 save_losses = False,   # save train/val losses
                 save_checkpoint = False,
                 save_checkpoint_steps = None,
                 run_dir = None
                 ):
        
        self.data_train = data_train 
        self.data_eval = data_eval
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_losses = save_losses
        self.save_checkpoint = save_checkpoint
        self.save_checkpoint_steps = save_checkpoint_steps
        self.run_dir = run_dir

        self.hyperparams = {
                                'num_epochs'    :num_epochs,
                                'batch_size'    :batch_size,
                                'eval_steps'    :eval_steps,
                                'learning_rate' :learning_rate,
                                'weight_decay'  :weight_decay
                                }

    def train(self, 
              model, 
              from_checkpoint = False, 
              from_checkpoint_run_dir = None, 
              from_checkpoint_step = None,
              lr_schedule = 'const',
              cosine_final_lr = 1e-6,
              cosine_T_max = 10,
              cosine_T_mult = 1,
              cosine_lr_restart_decay = 1
              ):
        
     


        if from_checkpoint:
            state_dict_path = os.path.join(from_checkpoint_run_dir, f'checkpoint_{from_checkpoint_step}')
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)

            losses_path = os.path.join(from_checkpoint_run_dir, 'losses.json')
            with open(losses_path, 'r') as f:
                losses = json.load(f)
            train_losses, eval_losses, steps, best_eval_loss = (losses['train_losses'], losses['eval_losses'], 
                                                                losses['steps'], losses['best_eval_loss'])
        
        else:
            train_losses, eval_losses, steps = [], [], []
            best_eval_loss = (float('inf'),0) # (loss, step)
            losses = {'train_losses': train_losses, 'eval_losses': eval_losses, 'best_eval_loss': best_eval_loss, 'steps': steps}


        #----------optimizer-----------------------
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        
        #----------learning rate schedule----------
        if lr_schedule == 'cosine':
            schedule = CosineScheduleLR(optimizer, final_lr = cosine_final_lr, T_max = cosine_T_max)
        elif lr_schedule == 'cosine_warm_restarts':
            schedule = CosineScheduleWarmRestarts(optimizer, 
                                                  final_lr = cosine_final_lr, 
                                                  T_max = cosine_T_max, 
                                                  T_mult = cosine_T_mult, 
                                                  lr_restart_decay = cosine_lr_restart_decay)

        
        lrs = []


        if self.save_losses or self.save_checkpoint:
            assert self.run_dir is not None, "Argument 'run_dir' must be provided to save losses or checkpoints"
            
            # create directory for logging, if it doesn't exist
            if not os.path.exists(self.run_dir):
                os.makedirs(self.run_dir)
            
            if self.save_losses:
                losses_path = os.path.join(self.run_dir, 'losses.json')
            
            if self.save_checkpoint:
                assert self.save_checkpoint_steps is not None, "Argument 'save_checkpoint_steps' must be provided when save_checkpoint = True"
            
            # save training hyperparameters
            hyperparams_path = os.path.join(self.run_dir,'hyperparameters.json')
            with open(hyperparams_path, 'w') as f:
                json.dump(self.hyperparams,f)

            # save model hyperparameters
            model_hyperparams = {  
                                'vocab_size': model.vocab_size,
                                'n_blocks'   : model.n_blocks,
                                'n_emb'     : model.n_emb,
                                'n_heads'   : model.n_heads,
                                'block_size': model.block_size,
                                'dropout'   : model.p
                                }
            model_hyperparams_path = os.path.join(self.run_dir,'model_hyperparameters.json')
            with open(model_hyperparams_path, 'w') as f:
                json.dump(model_hyperparams, f)



        # Data and training        
        
        dataloader = utils.DataLoader(self.data_train, self.batch_size)
        dataloader_eval = utils.DataLoader(self.data_eval, self.batch_size)
        

        lr_curr = self.learning_rate
        for epoch in tqdm(range(self.num_epochs)):
            train_loss = 0

            for batch in tqdm(dataloader, leave = False):
                
                # forward/backward/update
                logits, loss = model(batch[:,0,:], targets = batch[:,1,:])
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()

                # logging
                train_loss += loss.item()


            # train loss
            train_loss /= len(dataloader)
            train_losses.append(train_loss)

            # global step
            global_step = (epoch+1) * len(dataloader) 
            if from_checkpoint:
                global_step += steps[-1]
            steps.append(global_step)
            

        
            # Compute eval loss
            model.eval()
            eval_loss = 0
            for batch_eval in tqdm(dataloader_eval, leave = False):
                with torch.no_grad():
                    _, curr_eval_loss = model(batch_eval[:,0,:], targets = batch_eval[:,1,:])
                eval_loss += curr_eval_loss.item()
            
            eval_loss /= len(dataloader_eval)
            eval_losses.append(eval_loss)
            model.train()

            print(f'Epoch: {epoch+1:>4}/{self.num_epochs} || Train loss: {train_loss:.4f} || Eval loss: {eval_loss:.4f} || lr: {lr_curr:>5.4e}')

            # update best eval loss
            best_eval_loss = min( best_eval_loss, (eval_loss, global_step), key = lambda x:x[0] )
            losses['best_eval_loss'] = best_eval_loss
            
            # save losses 
            if self.save_losses:
                with open(losses_path, 'w') as f:
                    json.dump(losses, f)

            # reinit train loss
            train_loss = 0
            
            # save state_dict
            if self.save_checkpoint:
                path = os.path.join(self.run_dir, f'checkpoint_{epoch+1}')
                torch.save(model.state_dict(), path)


            #------------------------------------
            # lr schedule step
            lrs.append(lr_curr)
            if lr_schedule in ['cosine', 'cosine_warm_restarts']:
                schedule.step()
                lr_curr = optimizer.param_groups[0]['lr']
           
            
            lrs_path = os.path.join(self.run_dir, 'lrs.json')
            with open(lrs_path, 'w') as f:
                json.dump(lrs, f)


        return train_losses, eval_losses