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
                generator,
                emotion_proc,
                disc_word_len,
                train_loader,
                val_loader):        
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.generator = generator
        self.disc_word_len = disc_word_len
        self.emotion_proc = emotion_proc
        
        # self.run_name = 'GAN_class' + str(self.args.emo_dim) + '_lrg' + str(args.lr_g) + '_lrd' + str(args.lr_dsc) + '_WassFIX_NOPad_Att_Clamp1_Noise10_1by5_Text_ADH_batch8'
        self.run_name = 'GAN_old_WassSignFIX_NOPad_clamp5_Noise10_1by5_AFH_' + 'lrd' + str(args.lr_dsc) + 'meanvarall'
        self.plotter = SummaryWriter('runs_clean/' + self.run_name) 
        
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.emo_loss = nn.CrossEntropyLoss(reduction='mean')
        self.BCE = nn.CrossEntropyLoss(reduction='mean')
        self.global_step = 0
        if self.args.criterion == 'BCE':
            self.criterion = self.GAN_BCELoss
        else:
            self.criterion = self.GAN_WasserteinLoss
    
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
        lr_list = [self.generator.module.opt.param_groups]
        if self.args.disc_word_len:
            lr_list.append(self.disc_word_len.module.opt.param_groups)
        cnt = 0
        for lr in lr_list:
            for param_group in lr:
                print('LR {}: {}'.format(cnt, param_group['lr']))
                cnt+=1

    def sign_loss(self, relative_word_length, gen_relative_word_length):
        cosine = torch.cosine_similarity(relative_word_length, gen_relative_word_length, dim=1)
        return torch.mean(1 - cosine)

    def saveNetworks(self, fold):
        torch.save(self.generator.state_dict(), os.path.join(self.args.out_path, fold, 'generator.pt'))
        torch.save(self.disc_word_len.state_dict(), os.path.join(self.args.out_path, fold, 'disc_world_len.pt'))
        print('Networks has been saved to {}'.format(fold))

    # BCE loss
    def GAN_BCELoss(self, logit, label):
        if label == 'real':
            target = torch.ones([self.args.batch_size, 1])
            return self.BCE(logit, target)
            # return torch.mean(-torch.log(logit))
        if label == 'fake':
            target = torch.zeros([self.args.batch_size, 1])
            return self.BCE(logit, target)
            # return torch.mean(-torch.log(1-logit))

    def GAN_WasserteinLoss(self, logit, label):
        if label == 'real':
            return -logit.mean()
        if label == 'fake':
            return logit.mean()

    def step_disc_wordlen(self, data, epoch):
        self.disc_word_len.train()
        gen_relative_word_length, relative_word_length, gen_emo, emo_label = data
        self.disc_word_len.module.opt.zero_grad()
        
        # print(gen_relative_word_length.shape, relative_word_length.shape)
        logit_fake = self.disc_word_len(emo_label, gen_relative_word_length)
        logit_real = self.disc_word_len(emo_label, relative_word_length)
        loss_fake = self.criterion(logit_fake, 'fake')
        loss_real = self.criterion(logit_real, 'real')
        loss = loss_fake + loss_real

        loss.backward()
        self.disc_word_len.module.opt.step()

        if self.criterion == self.GAN_WasserteinLoss:
            # Clip the gradients
            with torch.no_grad():
                for param in self.disc_word_len.parameters():
                    param.clamp_(-0.05, 0.05)
            # # Gradient penalty
            # gradpenalty = self.disc_word_len.module.compute_gp(gen_relative_word_length, relative_word_length, emo_label)
            # loss += 5*gradpenalty


        wdistance = -(loss_fake + loss_real).item()
        losslst = np.array([loss_fake.item(), loss_real.item(), loss.item(), wdistance])
        return losslst
  
  
    def step_generator(self, data):
        self.disc_word_len.eval()

        self.generator.train()
        relative_word_length, emo_label, emotions_vec, pos_vec, people_vec = data
        self.generator.module.opt.zero_grad()
        gen_emotion, gen_relative_word_length  = self.generator(emotions_vec, pos_vec, people_vec)
        
        df = self.disc_word_len.forward(emotions_vec, gen_relative_word_length)
        gan_loss = self.criterion(df, 'real')
        recon_loss = self.mse_loss(relative_word_length, gen_relative_word_length)
        recon_mean_loss = self.mse_loss(relative_word_length.mean(dim=1), gen_relative_word_length.mean(dim=1))
        recon_var_loss = self.mse_loss(relative_word_length.var(dim=1), gen_relative_word_length.var(dim=1))
        recon_loss = recon_loss + recon_mean_loss + recon_var_loss
        sign_loss = self.sign_loss(relative_word_length, gen_relative_word_length)
        # emo_loss = self.emo_loss(gen_emotion, emo_label)
        
        loss =  gan_loss + recon_loss #+ 0.5*sign_loss # + 0.1*emo_loss
        loss.backward()
        self.generator.module.opt.step()

        # if np.random.random() > 0.995:
        #     print(np.round(gen_relative_word_length[:2,...].tolist(), 4), np.round(relative_word_length[:2,...].tolist(), 4), flush=True)

        losslst = np.array([recon_loss.item(),  sign_loss.item(), gan_loss.item(), loss.item()])
        return losslst

    
    # def train(self):
    #     update_batch_size = 8
    #     for epoch in tqdm(range(self.args.num_epochs)):
    #         gen_losses = np.array([0.,0.])
    #         dsc_losses = np.array([0.,0.,0.])
    #         self.disc_word_len.module.opt.zero_grad()
    #         self.generator.module.opt.zero_grad()

    #         diterator = iter(self.train_loader)
    #         for t in range(len(self.train_loader)):               
    #             relative_word_length, emo_label, emotions_vec, pos_vec, people_vec = [d.float().to(self.args.device) for d in next(diterator)]
    #             with torch.no_grad():
    #                 gen_emotion, gen_relative_word_length = self.generator(emotions_vec, pos_vec, people_vec)
    #             loss_fake = torch.mean(self.disc_word_len(emotions_vec, gen_relative_word_length))
    #             loss_real = -torch.mean(self.disc_word_len(emotions_vec, relative_word_length))
    #             loss = (loss_fake + loss_real)
    #             loss /= update_batch_size
    #             loss.backward()
    #             dsc_losses += np.array([loss.item(), loss_fake.item(), loss_real.item()])
                
    #             if t%5 == 0:
    #                 gen_emotion, gen_relative_word_length  = self.generator(emotions_vec, pos_vec, people_vec)
    #                 gan_loss = -torch.mean(self.disc_word_len.forward(emotions_vec, gen_relative_word_length))
    #                 recon_loss = self.mse_loss(relative_word_length, gen_relative_word_length)
    #                 loss = gan_loss #+ recon_loss 
    #                 loss /= (update_batch_size/5)
    #                 loss.backward()
    #                 gen_losses += np.array([recon_loss.item(), gan_loss.item()])

    #             if t%update_batch_size == 0:
    #                 self.disc_word_len.module.opt.step()
    #                 self.disc_word_len.module.opt.zero_grad()
    #                 self.generator.module.opt.step()
    #                 self.generator.module.opt.zero_grad()
    #                 with torch.no_grad():
    #                     for param in self.disc_word_len.parameters():
    #                         param.clamp_(-0.1, 0.1)
                
    #         length = len(self.train_loader)
    #         self.plotter.add_scalar("lossgen/recons", gen_losses[0]/(length/5), epoch)
    #         self.plotter.add_scalar("lossgen/", gen_losses[1]/(length/5), epoch)

    #         self.plotter.add_scalar("lossdisc/", dsc_losses[0]/length, epoch)
    #         self.plotter.add_scalar("lossdisc/fake", dsc_losses[1]/length, epoch)
    #         self.plotter.add_scalar("lossdisc/real", dsc_losses[2]/length, epoch)

    #         # if epoch%200 == 0:
    #         #     out_file = self.run_name + '/'
    #         #     if not os.path.exists(out_file):
    #         #         os.path.mkdirs(out_file)
    #         #     out_file = out_file + str(epoch) + '.txt'
    #         #     viterator = iter(self.val_loader)
    #         #     with open(out_file, 'w') as f:
    #         #         for t in range(len(self.val_loader)):
    #         #             relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec = [d.float().to(self.args.device) for d in next(viterator)]
    #         #             data = [relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec]
    #         #             with torch.no_grad():
    #         #                     gen_emotion, gen_relative_word_length = self.generator(emotions_vec, pos_vec, people_vec)
    #         #                     f.write('\t'.join([str(emotion_label.item()),
    #         #                         str(relative_word_length.tolist()[0]),
    #         #                         str(gen_relative_word_length.tolist()[0])])+'\n')
    #         #     f.close()
        
    #     self.displayLRs()
    #     self.saveNetworks('')
    #     self.plotter.flush()
    #     self.plotter.close()


    def train(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            gen_losses = np.array([0.,0.,0.,0.])
            dsc_losses = np.array([0.,0.,0.,0.])
            diterator = iter(self.train_loader)
            for t in range(len(self.train_loader)):               
                relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec = [d.float().to(self.args.device) for d in next(diterator)]
                data = [relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec]
                
                if epoch > 10 and self.global_step%7 == 0:
                    gen_losses += self.step_generator(data)
                else:
                    with torch.no_grad():
                        gen_emotion, gen_relative_word_length = self.generator(emotions_vec, pos_vec, people_vec)
                    data = [gen_relative_word_length, relative_word_length, gen_emotion, emotions_vec]
                    dsc_losses += self.step_disc_wordlen(data, epoch)
                self.global_step += 1
            
            # self.schdulerStep()

            length = len(self.train_loader)
            self.plotter.add_scalar("lossgen/recons", gen_losses[0]/length, epoch)
            # self.plotter.add_scalar("lossgen/emo", gen_losses[1]/length, epoch)
            # self.plotter.add_scalar("lossgen/sign", gen_losses[2]/length, epoch)
            self.plotter.add_scalar("lossgen/real", gen_losses[2]/length, epoch)
            self.plotter.add_scalar("lossgen/", gen_losses[3]/length, epoch)

            self.plotter.add_scalar("lossdisc/fake", dsc_losses[0]/length, epoch)
            self.plotter.add_scalar("lossdisc/real", dsc_losses[1]/length, epoch)
            self.plotter.add_scalar("lossdisc/", dsc_losses[2]/length, epoch)
            self.plotter.add_scalar("lossdisc/wdistance", dsc_losses[3]/length, epoch)

            if (epoch+1)%200 == 0:
                out_file = self.run_name + str(epoch) + '.txt'
                viterator = iter(self.train_loader)
                with open(out_file, 'w') as f:
                    for t in range(len(self.train_loader)):
                        relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec = [d.float().to(self.args.device) for d in next(viterator)]
                        data = [relative_word_length, emotion_label, emotions_vec, pos_vec, people_vec]
                        with torch.no_grad():
                                gen_emotion, gen_relative_word_length = self.generator(emotions_vec, pos_vec, people_vec)
                                f.write('\t'.join([str(emotion_label.item()),
                                    str(relative_word_length.tolist()[0]),
                                    str(gen_relative_word_length.tolist()[0])])+'\n')
                self.saveNetworks('')
                f.close()
        
        self.displayLRs()
        
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
                        _, gen_relative_word_length = self.generator(emotions_vec, pos_vec, people_vec)
                        f.write('\t'.join([str(emotion_label.item()), script[0],
                            str(relative_word_length.tolist()[0]),
                            str(gen_relative_word_length.tolist()[0])])+'\n')
        f.close()