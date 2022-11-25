from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class emoTrainer:
    def __init__(self, 
                args, 
                autoencoder,
                imle,
                train_loader,
                val_loader):        
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.autoencoder = autoencoder
        self.imle = imle
        
        self.run_name = 'IMLE_run_var_08'
        self.plotter = SummaryWriter('runs/' + self.run_name) 
        
        self.L2loss = torch.nn.MSELoss(reduction='mean')
        self.L1loss = torch.nn.L1Loss(reduction='mean')
    
    def freezeNet(self, network):
        for p in network.parameters():
            p.requires_grad = False
    
    def unfreezeNet(self, network):
        for p in network.parameters():
            p.requires_grad = True

    def schdulerStep(self):
        self.generator.module.scheduler.step()
        self.disc_word_len.module.scheduler.step()

    def displayLRs(self):
        lr_list = [self.autoencoder.module.opt.param_groups]
        
        cnt = 0
        for lr in lr_list:
            for param_group in lr:
                print('LR {}: {}'.format(cnt, param_group['lr']))
                cnt+=1

    def saveNetworks(self):
        if self.args.pre_train:
            out_file = os.path.join(self.args.out_path, self.run_name + 'autoencoder.pt')
            torch.save(self.autoencoder.state_dict(), out_file)
        else:
            out_file = os.path.join(self.args.out_path, self.run_name + '_IMLE.pt')
            torch.save(self.imle.state_dict(), out_file)
        print('Networks has been saved')

    def step_generator(self, data):
        self.disc_word_len.eval()
        self.generator.train()
        relative_word_length, emo_label, emotions_vec, pos_vec, people_vec = data
        self.generator.module.opt.zero_grad()
        gen_emotion, gen_relative_word_length  = self.generator(emotions_vec, pos_vec, people_vec)
        
        # print(gen_relative_word_length.shape, relative_word_length.shape)
        df = self.disc_word_len.forward(emotions_vec, gen_relative_word_length)
        gan_loss = self.criterion(df, 'fake')

        loss = self.mse_loss(relative_word_length, gen_relative_word_length)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.)
        self.generator.module.opt.step()

        if np.random.random() > 0.999:
            print(np.round(gen_relative_word_length[:2,...].tolist(), 4), np.round(relative_word_length[:2,...].tolist(), 4), flush=True)

        losslst = np.array([loss.item(),  gan_loss.item(), loss.item()])
        return losslst

    
    def step_autoencoder(self, data):
        self.autoencoder.train()
        relative_word_length, emo_label, emotions_vec, pos_vec, people_vec = data
        self.autoencoder.module.opt.zero_grad()
        gen_relative_word_length = self.autoencoder(emotions_vec, pos_vec, people_vec)

        loss = self.L2loss(gen_relative_word_length, relative_word_length)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 3.)
        self.autoencoder.module.opt.step()

        if np.random.random() > 0.9999:
            print(np.round(gen_relative_word_length[:2,...].tolist(), 4), np.round(relative_word_length[:2,...].tolist(), 4), flush=True)

        return loss.item()
    
    
    def step_imle(self, relative_word_length, mindist_latent_code):
        self.imle.train()
        self.imle.module.opt.zero_grad()

        loss = self.L2loss(relative_word_length, mindist_latent_code)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.imle.parameters(), 3.)
        self.imle.module.opt.step()
        return loss.item()


    def pre_train(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            diterator = iter(self.train_loader)
            loss = 0
            for t in range(len(self.train_loader)):               
                relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec = [d.float().to(self.args.device) for d in next(diterator)]
                data = [relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec]
                
                loss += self.step_autoencoder(data)

            self.autoencoder.module.scheduler.step() 
            length = len(self.train_loader)
            self.plotter.add_scalar("lossAE/recons", loss/length, epoch)

            if (epoch+1)%200 == 0:
                self.displayLRs()
                self.saveNetworks()
                out_file = os.path.join(self.args.out_path, self.run_name + str(epoch) + '.txt')
                viterator = iter(self.val_loader)
                with open(out_file, 'w') as f:
                    for t in range(len(self.val_loader)):
                        relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec, _ = [d.float().to(self.args.device) for d in next(viterator)]
                        data = [relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec]
                        with torch.no_grad():
                                gen_relative_word_length = self.autoencoder(emotions_vec, pos_vec, people_vec)
                                f.write('\t'.join([str(emotion_label.item()),
                                    str(relative_word_length.tolist()[0]),
                                    str(gen_relative_word_length.tolist()[0])])+'\n')
                f.close()
                

    def collect_variance(self):
        diterator = iter(self.train_loader)
        encoded_vecs = []
        for t in range(len(self.train_loader)):  
            _, _, emotions_vec, pos_vec, people_vec = [d.float().to(self.args.device) for d in next(diterator)]
            
            self.imle.eval()
            z = self.imle.module.encode(emotions_vec, pos_vec, people_vec)
            encoded_vecs.append(z.std())
        encoded_vecs = torch.stack(encoded_vecs)
        print(encoded_vecs.mean(), encoded_vecs.var())

        
    def train(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            diterator = iter(self.train_loader)
            loss = 0
            for t in range(len(self.train_loader)):  
                             
                relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec = [d.float().to(self.args.device) for d in next(diterator)]
                
                self.imle.train()
                z = self.imle.module.encode(emotions_vec, pos_vec, people_vec)
                mindist_latent_code = self.imle.module.get_mindist_latent_codes(z, 16, self.args.batch_size, [z.shape[1], z.shape[2]], relative_word_length)
                loss += self.step_imle(relative_word_length, mindist_latent_code)
                
                if np.random.random() > 0.9999:
                    print(np.round(mindist_latent_code[:2,...].tolist(), 4), np.round(relative_word_length[:2,...].tolist(), 4), flush=True)
                
            length = len(self.train_loader)
            self.plotter.add_scalar("lossIMLE/generate", loss/length, epoch)

            if epoch%50==0:
                self.saveNetworks()

        self.plotter.flush()
        self.plotter.close()

        
    def test(self):
        diterator = iter(self.val_loader)
        out_file = os.path.join(self.args.out_path, self.run_name + '_3samples_test.txt')
        with open(out_file, 'w') as f:
            for _ in range(len(self.val_loader)):  
                relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec, script = [d for d in next(diterator)]
                self.imle.eval()
                with torch.no_grad():
                    for _ in range(3):
                        z = self.imle.module.encode(emotions_vec.float(), pos_vec.float(), people_vec.float())
                        gen_relative_word_length = self.imle.module.get_single_latent_code(z, [z.shape[1], z.shape[2]])
                        f.write('\t'.join([str(emotion_label.item()), script[0],
                                        str(relative_word_length.tolist()[0]),
                                        str(gen_relative_word_length.tolist()[0])])+'\n')
        f.close()