import torch 
import torch.nn as nn
import torch.nn.functional as F

class STFTLayer(nn.Module):
    def __init__(self, n_fft=512, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
        
    def forward(self, x):
        # Convert (B, T) to (B, 2, F, T)
        spec = torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        spec = torch.view_as_real(spec)
        return spec.permute(0, 3, 1, 2)

class ISTFTLayer(nn.Module):
    def __init__(self, n_fft=512, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
        
    def forward(self, real, imag):
        assert real.shape == imag.shape, "real and imag must have the same shape"
        spec = torch.stack([real, imag], -1)
        spec_complex = torch.view_as_complex(spec.contiguous())
        return torch.istft(spec_complex, self.n_fft, self.hop_length, window=self.window)

class Encoder(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=(3,5), 
                              stride=(2,2), padding=(1,2))
        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=(3,5), 
                              stride=(2,2), padding=(1,2))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=(3,5), 
                                     stride=(2,2), padding=(1,2), output_padding=(0,1))
        self.up2 = nn.ConvTranspose2d(channels, 2, kernel_size=(3,2), 
                                     stride=(2,2), padding=(1,2), output_padding=(0,1))
        
    def forward(self, x):
        x = F.relu(self.up1(x))
        x = self.up2(x)
        return x

class BiGRUReconstructor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, in_dim)

    def forward(self, x):
        B, C, F, T = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * F, C, T).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1).reshape(B, F, C, T).permute(0, 2, 1, 3)
        return x
    
class AudioEnhancer(nn.Module):
    def __init__(self, n_fft=512, hop=160, channels=32):
        super().__init__()
        self.stft = STFTLayer(n_fft, hop)
        self.encoder = Encoder(2, channels)
        self.rnn = BiGRUReconstructor(64, 64)
        self.decoder = Decoder(channels)
        self.istft = ISTFTLayer(n_fft, hop)
        
    def forward(self, x):
        spec = self.stft(x)
        enc = self.encoder(spec)
        rnn_out = self.rnn(enc)
        dec = self.decoder(rnn_out)
        real = dec[:,0] * spec[:,0] - dec[:,1] * spec[:,1]
        imag = dec[:,0] * spec[:,1] + dec[:,1] * spec[:,0]
        out = self.istft(real, imag)
        return out

if __name__ == "__main__":
    model = AudioEnhancer(n_fft=512, hop=160).eval()
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print(flops, params)
    
    x = torch.randn(2, 16000)
    y = model(x)
    print(y.shape)
