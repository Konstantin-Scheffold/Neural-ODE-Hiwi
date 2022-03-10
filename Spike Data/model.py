import torch
import torch.nn as nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.activation = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden * 4)
        self.fc3 = nn.Linear(nhidden * 4, nhidden* 4)
        self.fc4 = nn.Linear(nhidden * 4, nhidden * 4)
        self.fc5 = nn.Linear(nhidden * 4, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        out = self.activation(out)
        out = self.fc5(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        dim = obs_dim + nhidden
        self.conv_a = nn.Conv1d(1, dim, kernel_size=16, stride=1, padding=7)
        self.conv_b = nn.Conv1d(dim, dim, kernel_size=16, stride=2, padding=7)
        self.conv_c = nn.Conv1d(dim, 1, kernel_size=16, stride=2, padding=7)
        self.conv_out = nn.ELU(inplace=True)

        self.encoder_1 = nn.Linear(25, nhidden)
        self.encoder_2 = nn.Linear(nhidden, latent_dim * 2)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x, h):
        x = x.reshape((self.nbatch, 1, -1))
        h = h.reshape((self.nbatch, 1, -1))

        combined = torch.cat((x, h), dim=2)

        Conv_A = self.conv_a(combined)
        Conv_B = self.conv_b(Conv_A)
        Conv_C = self.conv_c(Conv_B)
        Conv_out = self.conv_out(Conv_C)

        Enc_1 = self.encoder_1(Conv_out)
        Enc_1 = self.activation(Enc_1)
        Enc_2 = self.encoder_2(Enc_1)
        Enc_2 = self.activation(Enc_2)

        Enc_1 = torch.squeeze(Enc_1, dim=1)
        Enc_2 = torch.squeeze(Enc_2, dim=1)
        return Enc_2, Enc_1

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, time_points, section, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.activation = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)
        self.section = section
        self.deconv_a = nn.ConvTranspose1d(time_points, time_points, kernel_size=16, stride=1, padding=7)
        self.deconv_b = nn.ConvTranspose1d(time_points, time_points, kernel_size=16, stride=2, padding=7)
        self.deconv_out = nn.Tanh()

    def forward(self, z):
        Fc1 = self.activation(self.fc1(z))
        Fc2 = self.activation(self.fc2(Fc1))

        Deconv_a = self.deconv_a(Fc2)
        Deconv_b = self.deconv_b(Deconv_a)
        out = self.deconv_out(Deconv_b) * self.section

        return out