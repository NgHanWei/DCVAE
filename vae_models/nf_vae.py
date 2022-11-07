from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .flow.Flow import *

class NFVAE_LSTM(nn.Module):
    def init_hidden(self, batch_size, hidden_size, num_directions, num_layers):
        return (torch.autograd.Variable(torch.zeros(num_directions*num_layers, batch_size, hidden_size)).cuda(),
                torch.autograd.Variable(torch.zeros(num_directions*num_layers, batch_size, hidden_size)).cuda())

    def __init__(self, filters, channels, features, data_type, data_length):
        super(NFVAE_LSTM, self).__init__()
 
        self.filters = filters
        self.channels = channels
        self.features = features
        self.data_type = data_type
        self.data_length = data_length

        self.flow = Flow(features, 'planar', 8)

        # encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
            # nn.LSTM(50,filters*2,1,bidirectional=False),
            nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )

        self.lstm = nn.LSTM(16, 16, 1, batch_first = True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden = self.init_hidden(16, 16, 1, 1)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=filters*2,out_channels=filters*4, kernel_size=(channels,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
            nn.LeakyReLU(0.2)
        )

        self.flatten =  nn.Flatten()

        if data_type == 'eeg':
            dense_var = 156
            dense_var_2 = 156
        else:
            dense_var = 236
            dense_var_2 = 240

        self.dense = nn.Linear(dense_var*filters,32)
        self.dense2 = nn.Linear(features,32)
        self.dense3 = nn.Linear(32,dense_var_2*filters)
        self.dense_features = nn.Linear(32,features)
        # decoder

        self.decode_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*4, kernel_size=(channels,1),stride=(1,1)),
            nn.BatchNorm2d(num_features=filters*4, eps=1e-03, momentum=0.99 ),
        )

        self.lstm_2 = nn.LSTM(32, 32, 1, batch_first = True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_2 = self.init_hidden(16, 32, 1, 1)

        if data_type == 'eeg':
            self.decode_layer2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*2, kernel_size=(1,50),stride=(1,25)),
                nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),            
            )
        else:
            self.decode_layer2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=filters*4,out_channels=filters*2, kernel_size=(1,25),stride=(1,25)),
                nn.BatchNorm2d(num_features=filters*2, eps=1e-03, momentum=0.99 ),            
            )

        self.output = nn.ConvTranspose2d(in_channels=filters*2,out_channels=1,kernel_size=5,padding=(2,2))


    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        sample = self.flow(sample)
        return sample

    def forward(self, x):
        # encoding
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x, (hn,cn) = self.lstm(x,self.hidden)
        # print(x.shape)
        x = x.view(x.size(0),62,39,-1)
        # print(x.shape)
        x = x.permute(0,3,1,2)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = F.relu(self.dense(x))
        # print(x.shape)

        # get `mu` and `log_var`
        mu = self.dense_features(x)
        log_var = self.dense_features(x) + 1e-8
        # print(mu.shape)
        # print(log_var.shape)
        # get the latent vector through reparameterization
        z , log_det = self.reparameterize(mu, log_var)
        # print(z.shape)

        # decoding
        x = F.relu(self.dense2(z))
        x = F.relu(self.dense3(x))
        if self.data_type == 'eeg':
            view_var = 39
        else:
            view_var = 60 * len(x)
        x = x.view(-1, self.filters*4, 1, view_var)
        # print(x.shape)

        x = self.decode_layer1(x)
        # print(x.shape)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x, (hn,cn) = self.lstm_2(x,self.hidden_2)
        # print(x.shape)
        x = x.view(x.size(0),62,39,-1)
        # print(x.shape)
        x = x.permute(0,3,1,2)

        x = self.decode_layer2(x)
        # print(x.shape)
        reconstruction = self.output(x)
        # print(reconstruction.shape)

        # decoding
        return reconstruction, mu, log_var, log_det