import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import normal
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import numpy as np
import math
from train import initParams


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class SequenceWise2d(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise2d, self).__init__()
        self.module = module

    def forward(self, x):
        n, t, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.contiguous().view(t * n, c, w, h)
        x = self.module(x)
        _, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.view(n, t, c, w, h)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


# BxNx32
class TEXTENCODER(nn.Module):
    def __init__(self, args, bnorm=True):
        super(TEXTENCODER, self).__init__() 
        self.args = args

        in_dim = 54
        self.fc_0 = nn.Sequential(nn.Linear(in_dim, 16),
                        nn.Tanh())
        self.lstm = nn.LSTM(input_size=16, hidden_size=self.args.text_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.FCN = nn.Sequential(nn.Linear(self.args.text_dim*2, self.args.text_dim),
                                nn.Tanh())
        self.fc_1 = nn.Sequential(nn.Linear(in_dim, 16))

    def forward(self, x):
        x = self.fc_0(x)
        x, _ = self.lstm(x)
        out = self.FCN(x)
        return out


# BxNx1
class NOISEGENERATOR(nn.Module):
    def __init__(self, args, debug=False):
        super(NOISEGENERATOR, self).__init__()
        self.args = args
        self.noise = normal.Normal(0, 0.7)
        self.noise_imle = normal.Normal(0, 0.8)
        self.noise_rnn = nn.LSTM(self.args.noise_dim, self.args.noise_dim, num_layers=1, batch_first=True)

    def forward(self, z_spch):
        self.noise_rnn.flatten_parameters()
        noise = []
        for i in range(z_spch.size(0)):
            noise.append(self.noise.sample((z_spch.size(1), z_spch.size(2))))
        noise = torch.stack(noise, 0).to(self.args.device)
        return noise

    def get_latent_codes(self, batch_size, shape):
        noise = []
        for i in range(batch_size):
            noise.append(self.noise_imle.sample((shape[0], shape[1])))
        noise = torch.stack(noise, 0).to(self.args.device)
        return noise


# BxNx6
class EMOTIONPROCESSOR(nn.Module):
    def __init__(self, args, debug=False):
        super(EMOTIONPROCESSOR, self).__init__()
        self.args = args
        self.fc_1 = nn.Sequential(
            nn.Linear(3, self.args.emo_dim),
            nn.ReLU(),
            nn.Linear(self.args.emo_dim, self.args.emo_dim)
        )
    def forward(self, emotion_cond):
        emotion_cond = self.fc_1(emotion_cond)
        return emotion_cond
    

# BxNx10
class PEOPLEPROCESSOR(nn.Module):
    def __init__(self, args, debug=False):
        super(PEOPLEPROCESSOR, self).__init__()
        self.args = args
        self.fc_1 = nn.Sequential(
            nn.Linear(91, 32),
            nn.ReLU(),
            nn.Linear(32, self.args.people_dim)
        )
    def forward(self, people_cond):
        people_cond = self.fc_1(people_cond)
        return people_cond


class DECODER(nn.Module):
    def __init__(self, args, debug=False):
        super(DECODER, self).__init__()
        self.debug = debug
        self.args = args
        self.drp_rate = 0
        
        self.fc_1 = nn.Sequential(
            nn.Linear(self.args.text_dim*2+self.args.emo_dim, self.args.filters[-1]),
            nn.LeakyReLU(0.2),
        )

        self.lstm_1 = nn.LSTM(input_size=self.args.text_dim+self.args.emo_dim+self.args.people_dim+self.args.noise_dim//2, 
                              hidden_size=self.args.filters[0], num_layers=2, batch_first=True)
        self.lstm_2 = nn.Sequential(nn.Tanh(),
                                  nn.LSTM(input_size=self.args.filters[0], hidden_size=1, num_layers=1, batch_first=True))

        in_dim = self.args.text_dim+self.args.emo_dim+self.args.people_dim+self.args.noise_dim//2
        self.emo_classifier = nn.Sequential(
            nn.Linear(in_dim, self.args.emo_dim),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.tanh = nn.Tanh()

        self.fc_1 = nn.Sequential(nn.Linear(in_dim, 1),
                                nn.Tanh())
        
    def forward(self, encodings_vec):
        h, _ = self.lstm_1(encodings_vec)
        gen_word_len, _ = self.lstm_2(h)
        gen_word_len = self.tanh(gen_word_len).squeeze(dim=2)

        gen_emotion = self.emo_classifier(encodings_vec)
        return gen_emotion, gen_word_len


class GENERATOR(nn.Module):
    def __init__(self, args, train=True, debug=False):
        super(GENERATOR, self).__init__()
        self.args = args
        self.debug = debug
        self.text_encoder = TEXTENCODER(args)
        self.noise_generator = NOISEGENERATOR(args)
        self.emotion_processor = EMOTIONPROCESSOR(args)
        self.people_processor = PEOPLEPROCESSOR(args)

        self.decoder = DECODER(args)

        # self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_g, betas=(0.5, 0.9))
        self.opt = optim.RMSprop(list(self.parameters()), lr = self.args.lr_g)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.6, last_epoch=-1)

    def forward(self, emotion, pos_vec, people_vec):
        z_text = self.text_encoder(pos_vec)
        z_emo = self.emotion_processor(emotion)
        z_emo = z_emo.unsqueeze(1).repeat(1, z_text.size(1), 1)
        z_people = self.people_processor(people_vec)
        z_people = z_people.unsqueeze(1).repeat(1, z_text.size(1), 1)

        z = torch.cat((z_text, z_emo, z_people), 2)
        z_noise = self.noise_generator(z)

        # Concatenate noise
        z = torch.cat((z, z_noise), 2)

        # Add noise
        # z += z_noise
        
        gen_emotion, gen_word_len = self.decoder(z)
        # print("gen_word_len", gen_word_len.shape)
        if self.debug:
            return gen_word_len
        else:
            return gen_emotion, gen_word_len
        


class DISCWORDLEN(nn.Module):
    def __init__(self, args):
        super(DISCWORDLEN, self).__init__()
        self.args = args
        self.fc_1 = nn.Sequential(
            nn.Linear(3, 2),
            nn.LeakyReLU(0.2)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(4+2, 1),
        )
        if args.criterion == 'BCE':
            self.fc_2 = nn.Sequential(
            nn.Linear(4+2, 1),
            nn.Sigmoid()
        )
        # self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_dsc, betas=(0.5, 0.9))
        self.opt = optim.RMSprop(list(self.parameters()), lr = self.args.lr_dsc)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.6, last_epoch=-1) 
    
    def forward(self, emotion, relative_word_length):
        features = torch.tensor([torch.std(relative_word_length, dim=1), torch.max(relative_word_length, dim=1).values, torch.min(relative_word_length, dim=1).values, torch.mean(relative_word_length, dim=1)])
        features = features.unsqueeze(dim=0)
        emotion = self.fc_1(emotion)
        h_ = torch.cat((emotion, features), 1)
        h = self.fc_2(h_)
        return h
    
    def compute_gp(self, real_data, fake_data, emotion):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        
        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data
        interpolation.requires_grad_(True)
        # get logits for interpolated images
        interp_logits = self.forward(emotion, interpolation)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

class DISCEMO(nn.Module):
    def __init__(self, args, debug=False):
        super(DISCEMO, self).__init__()
        self.args = args
        self.drp_rate = 0
        self.filters = [(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2), (512, 3, 2)]
        prev_filters = 3
        for i, (num_filters, filter_size, stride) in enumerate(self.filters):
            setattr(self, 
                    'conv_'+str(i+1), 
                    nn.Sequential(
                    nn.Conv2d(prev_filters, num_filters, kernel_size=filter_size, stride=stride, padding=filter_size//2),
                    nn.LeakyReLU(0.3)
                )
            )
            prev_filters = num_filters

        self.projector = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.LeakyReLU(0.3),
            nn.Linear(2048, 512)
        )
        self.rnn_1 = nn.LSTM(512, 512, 1, bidirectional=False, batch_first=True)
        self.cls = nn.Sequential(
            nn.Linear(512, 6+1)
        )
        self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_emo, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.1, last_epoch=-1)

    def forward(self, condition, video):
        x = video
        n, t, w, h = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.contiguous().view(t * n, c, w, h)
        for i in range(len(self.filters)):
            x = getattr(self, 'conv_'+str(i+1))(x)
        h = x.view(n, t, -1)
        h = self.projector(h)
        h, _ = self.rnn_1(h)
        h_class = self.cls(h[:, -1, :])
        return h_class

    def enableGrad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad_(requires_grad)

    def compute_grad_penalty(self, video_gt, video_pd, image_c, classes):
        interpolated = video_gt.data #+ (1-alpha) * video_pd.data
        interpolated = Variable(interpolated, requires_grad=True)
        d_out_c = self.forward(image_c, interpolated)
        classes = torch.cat((classes, torch.zeros(classes.size(0), 1).to(self.args.device)), 1)
        
        grad_dout = torch.autograd.grad(
            outputs= d_out_c, 
            inputs= interpolated,
            grad_outputs= classes.to(self.args.device),
            create_graph=True, 
            retain_graph=True,
        )[0]
        grad_dout = grad_dout.contiguous().view(grad_dout.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(grad_dout ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean(), gradients_norm.mean()

      
class AUTOENCODER(nn.Module):
    def __init__(self, args, train=True, debug=False):
        super(AUTOENCODER, self).__init__()
        self.args = args
        self.debug = debug
        self.text_encoder = AE_TEXTENCODER(args)
        self.emotion_processor = EMOTIONPROCESSOR(args)
        self.people_processor = PEOPLEPROCESSOR(args)

        self.decoder = AE_DECODER(args)

        self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_g, betas=(0.8, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.5, last_epoch=-1)

    def forward(self, emotion, pos_vec, people_vec):
        z_text = self.text_encoder(pos_vec)
        z_emo = self.emotion_processor(emotion)
        z_emo = z_emo.unsqueeze(1).repeat(1, z_text.size(1), 1)
        z_people = self.people_processor(people_vec)
        z_people = z_people.unsqueeze(1).repeat(1, z_text.size(1), 1)
        
        z = torch.cat((z_text, z_emo, z_people), 2)
        gen_word_len = self.decoder(z)

        return gen_word_len
    
    
class IMLE(nn.Module):
    def __init__(self, args, autoencoder, train=True, debug=False):
        super(IMLE, self).__init__()
        self.args = args
        self.debug = debug
        self.autoencoder = autoencoder
        self.text_encoder = self.autoencoder.module.text_encoder
        self.emotion_processor = self.autoencoder.module.emotion_processor
        self.people_processor = self.autoencoder.module.people_processor
        self.decoder = self.autoencoder.module.decoder

        self.noise_generator = NOISEGENERATOR(args)
        
        self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_g, betas=(0.8, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.5, last_epoch=-1)

    def find_nearest_neighbor(self, x, out):
        dist = F.pairwise_distance(
            x.view(x.size(0), -1).unsqueeze(2), out.view(out.size(0), -1).t().unsqueeze(0))
        idx = dist.argmin(dim=1)
        return out[idx]
    
    def encode(self, emotion, pos_vec, people_vec):
        z_text = self.text_encoder(pos_vec)
        z_emo = self.emotion_processor(emotion)
        z_emo = z_emo.unsqueeze(1).repeat(1, z_text.size(1), 1)
        z_people = self.people_processor(people_vec)
        z_people = z_people.unsqueeze(1).repeat(1, z_text.size(1), 1)
        
        z = torch.cat((z_text, z_emo, z_people), 2)
        return z
    
    def get_mindist_latent_codes(self, z, n_samples, batch_size, shape, real_world_length):
        min_dist_y = None
        min_dist = torch.tensor(10e10)

        z_noise = self.noise_generator.get_latent_codes(n_samples, shape)
        for i in range(n_samples):
            z_tmp = z + z_noise[i, ...]
            gen_word_len_tmp = self.decoder(z_tmp)
            local_dist = F.pairwise_distance(gen_word_len_tmp, real_world_length)
            if local_dist < min_dist:
                min_dist_y = gen_word_len_tmp
                min_dist = local_dist
        return min_dist_y

    def get_single_latent_code(self, z, shape):
        z_noise = self.noise_generator.get_latent_codes(1, shape)
        z = z + z_noise[0]
        gen_word_len= self.decoder(z)
        return gen_word_len
    

class AE_DECODER(nn.Module):
    def __init__(self, args, debug=False):
        super(AE_DECODER, self).__init__()
        self.debug = debug
        self.args = args
        self.drp_rate = 0
        
        self.fc_1 = nn.Sequential(
            nn.Linear(self.args.text_dim*2+self.args.emo_dim, self.args.filters[-1]),
            nn.LeakyReLU(0.2),
        )
        self.tanh = nn.Tanh()
        self.lstm_1 = nn.LSTM(input_size=self.args.text_dim+self.args.emo_dim+self.args.people_dim,
                              hidden_size=self.args.filters[0], num_layers=1, batch_first=True)
        self.lstm_2 = nn.Sequential(nn.Tanh(),
                                  nn.LSTM(input_size=self.args.filters[0], hidden_size=1, num_layers=1, batch_first=True))
        
    def forward(self, encodings_vec):
        h, _ = self.lstm_1(encodings_vec)
        gen_word_len, _ = self.lstm_2(h)
        gen_word_len = self.tanh(gen_word_len).squeeze(dim=2)
        return gen_word_len


# BxNx32
class AE_TEXTENCODER(nn.Module):
    def __init__(self, args, bnorm=True):
        super(AE_TEXTENCODER, self).__init__() 
        self.args = args

        in_dim = 54
        self.fc_0 = nn.Sequential(nn.Linear(in_dim, 16), nn.Tanh())
        self.lstm = nn.LSTM(input_size=16, hidden_size=self.args.text_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.FCN = nn.Sequential(nn.Linear(self.args.text_dim*2, self.args.text_dim),
                                nn.Tanh())
        self.fc_1 = nn.Sequential(nn.Linear(in_dim, 16))

    def forward(self, x):
        x = self.fc_0(x)
        x, _ = self.lstm(x)
        out = self.FCN(x)
        return out
    
if __name__ == "__main__":
    args = initParams()
    TEXTENCODER(args)
