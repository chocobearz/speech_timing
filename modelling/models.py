import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import normal
import torch.optim as optim
from torch.autograd import Variable
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
        self.lstm = nn.LSTM(input_size=54, hidden_size=self.args.text_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.FCN = nn.Sequential(nn.Linear(self.args.text_dim*2, self.args.text_dim))

        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1))
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=54, num_heads=9)
        self.fc_1 = nn.Sequential(nn.Linear(54, 16))

    def forward(self, x):
        Q = K = V = x 
        attn_output, attn_output_weights = self.multihead_attn(Q, K, V)
        # print(attn_output.shape, x.shape)
        out = self.fc_1(attn_output)
        # x, _ = self.lstm(x)
        # x = x.unsqueeze(dim=1)
        # x = self.conv(x).squeeze(dim=1)
        # out = self.FCN(x)
        # out = self.leakyrelu(x)
        return out


# BxNx1
class NOISEGENERATOR(nn.Module):
    def __init__(self, args, debug=False):
        super(NOISEGENERATOR, self).__init__()
        self.args = args
        self.noise = normal.Normal(0, 0.7)
        self.noise_rnn = nn.LSTM(self.args.noise_dim, self.args.noise_dim, num_layers=1, batch_first=True)

    def forward(self, z_spch):
        self.noise_rnn.flatten_parameters()
        noise = []
        for i in range(z_spch.size(1)):
            noise.append(self.noise.sample((z_spch.size(0), self.args.noise_dim)))
        noise = torch.stack(noise, 1).to(self.args.device)
        # noise, _ = self.noise_rnn(noise)
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

        self.lstm_1 = nn.LSTM(input_size=self.args.text_dim*3+self.args.emo_dim, 
                              hidden_size=self.args.filters[0], num_layers=2, batch_first=True)
        self.lstm_2 = nn.Sequential(nn.Tanh(),
                                  nn.LSTM(input_size=self.args.filters[0], hidden_size=1, num_layers=1, batch_first=True))

        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=5, out_channels=1, kernel_size=3, stride=1, padding=1))
        self.FCN = nn.Sequential(nn.Linear(self.args.text_dim+self.args.emo_dim, self.args.filters[0]),
                                 nn.Linear(self.args.filters[0], self.args.MAX_LEN),
                                 nn.Tanh(),
                                 nn.Linear(self.args.MAX_LEN, 1))
        

        self.emo_classifier = nn.Sequential(
            nn.Linear(40, self.args.emo_dim),
            # nn.Linear(self.args.text_dim*3+self.args.emo_dim, self.args.emo_dim),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(self.args.emo_dim*self.args.MAX_LEN, self.args.emo_dim),
            nn.ReLU()
        )
        self.tanh = nn.Tanh()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=40, num_heads=8)
        self.fc_1 = nn.Sequential(nn.Linear(40, 8),
                                nn.ReLU(),
                                nn.Linear(8, 1))
        
    def forward(self, encodings_vec):
        # h, _ = self.lstm_1(encodings_vec)
        # gen_word_len, _ = self.lstm_2(h)
        # gen_word_len = self.tanh(gen_word_len).squeeze(dim=2)

        # encodings_vec = encodings_vec.unsqueeze(dim=1)
        # encodings_vec = self.conv(encodings_vec).squeeze(dim=1)

        # gen_word_len = self.FCN(encodings_vec).squeeze(dim=-1)

        Q = K = V = encodings_vec
        gen_word_len, _ = self.multihead_attn(Q, K, V)
        gen_word_len = self.fc_1(gen_word_len).squeeze(dim=-1)

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
        self.decoder = DECODER(args)

        # self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_g, betas=(0.5, 0.999))
        self.opt = optim.RMSprop(list(self.parameters()), lr = self.args.lr_g)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.6, last_epoch=-1)

    def forward(self, emotion, pos_vec):
        z_text = self.text_encoder(pos_vec)
        z_emo = self.emotion_processor(emotion)
        z_emo = z_emo.unsqueeze(1).repeat(1, z_text.size(1), 1)
        z = torch.cat((z_text, z_emo), 2)
        z_noise = self.noise_generator(z)

        # Concatenate noise
        z = torch.cat((z, z_noise), 2)

        # Add noise
        # z += z_noise
        
        gen_emotion, gen_word_len = self.decoder(z)

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
            nn.Linear(7+2, 1),
            # nn.Sigmoid()
        )
        # self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_dsc, betas=(0.7, 0.999))
        self.opt = optim.RMSprop(list(self.parameters()), lr = self.args.lr_dsc)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.6, last_epoch=-1) 
    
    def forward(self, emotion, relative_word_length):
        # features = torch.tensor([torch.std(relative_word_length, dim=1), torch.max(relative_word_length, dim=1), torch.min(relative_word_length, dim=1), torch.mean(relative_word_length, dim=1)])
        # features = features.unsqueeze(dim=0)
        
        
        emotion = self.fc_1(emotion)
        # print(emotion.shape, relative_word_length.shape)
        h_ = torch.cat((emotion, relative_word_length), 1)
        h = self.fc_2(h_)
        return h


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

    
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


if __name__ == "__main__":
    args = initParams()
    TEXTENCODER(args)