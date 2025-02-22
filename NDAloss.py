import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram, MelScale
import numpy as np
import webrtcvad

class SISNRLoss(nn.Module):
    def __init__(self):
        super(SISNRLoss, self).__init__()

    def forward(self, enhanced_signal, clean_signal):
        min_length = min(enhanced_signal.size(-1), clean_signal.size(-1))
        enhanced_signal = enhanced_signal[:, :min_length]
        clean_signal = clean_signal[:, :min_length]

        enhanced_mean = torch.mean(enhanced_signal, dim=-1, keepdim=True)
        clean_mean = torch.mean(clean_signal, dim=-1, keepdim=True)
        enhanced_signal = enhanced_signal - enhanced_mean
        clean_signal = clean_signal - clean_mean

        dot_product = torch.sum(enhanced_signal * clean_signal, dim=-1, keepdim=True)
        target_energy = torch.sum(clean_signal ** 2, dim=-1, keepdim=True)
        s_target = (dot_product / (target_energy + 1e-8)) * clean_signal

        e_noise = enhanced_signal - s_target

        s_target_energy = torch.sum(s_target ** 2, dim=-1)
        e_noise_energy = torch.sum(e_noise ** 2, dim=-1)
        si_snr = 10 * torch.log10(s_target_energy / (e_noise_energy + 1e-8))

        return -torch.mean(si_snr)

# refer https://github.com/wiseman/py-webrtcvad/blob/e283ca41df3a84b0e87fb1f5cb9b21580a286b09/example.py#L63
def vad_collector(sp, fs, frame_size, vad):
    """Filters out non-voiced audio frames.

    Arguments:
    sp - PCM audio data as a numpy array (int16 format).
    fs - The audio sample rate, in Hz.
    frame_size - The number of samples in each frame (must be 160, 320, or 480 for 10ms, 20ms, 30ms).
    vad - An instance of webrtcvad.Vad.

    Returns: A numpy array indicating whether each frame contains speech (1 for speech, 0 for non-speech).
    """
    frame_shift = frame_size // 2
    num_frames = (len(sp) - frame_size) // frame_shift + 1

    vad_results = []

    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_size
        frame = sp[start:end]

        if len(frame) != frame_size:
            break  

        frame_bytes = frame.tobytes()
        is_speech = vad.is_speech(frame_bytes, fs)
        vad_results.append(1 if is_speech else 0)

    vad_results = [0] + vad_results + [0]

    return np.array(vad_results, dtype=np.float32)

class NDALoss(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=320, hop_length=160, n_mels=80, gamma=5, mu=20, omega=20, eta=3, tau=2):
        super(NDALoss, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.gamma = gamma
        self.mu = mu
        self.omega = omega
        self.eta = eta
        self.tau = tau

        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1)
        self.mel_scale = MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1)

    def to(self, device):
        super().to(device)
        self.spectrogram = self.spectrogram.to(device)
        self.mel_scale = self.mel_scale.to(device)
        return self

    def compute_mel_spectrum(self, waveform):
        spectrogram = self.spectrogram(waveform)
        mel_spectrum = self.mel_scale(spectrogram)
        return torch.pow(mel_spectrum, 1/3)

    def compute_differential_spectrum(self, mel_spectrum):
        alpha = torch.tensor([0.25, 0.5, 0.75, 1.0], device=mel_spectrum.device)
        velocity_diff = torch.zeros_like(mel_spectrum)
        for k in range(1, 5):
            velocity_diff[:, :, :-k] += alpha[k-1] * (mel_spectrum[:, :, k:] - mel_spectrum[:, :, :-k])

        beta = torch.tensor([-0.357193, -0.607143, -0.285714, 0.25, 1.0], device=mel_spectrum.device)
        acceleration_diff = torch.zeros_like(mel_spectrum)
        for k in range(5):
            if k == 0:
                acceleration_diff += beta[k] * mel_spectrum
            else:
                acceleration_diff[:, :, :-k] += beta[k] * (mel_spectrum[:, :, k:] - mel_spectrum[:, :, :-k])

        return velocity_diff, acceleration_diff

    def envelope_loss(self, B_c, B_s, V):
        diff = B_c - B_s
        asym_diff = torch.where(diff > 0, diff, self.eta * diff)
        loss = torch.mean(torch.abs(asym_diff * V))
        return loss

    def continuity_loss_speech(self, B_vel_c, B_vel_s, B_acc_c, B_acc_s, V):
        loss_vel = torch.mean(torch.abs(B_vel_c - B_vel_s) * V)
        loss_acc = torch.mean(torch.abs(B_acc_c - B_acc_s) * V)
        return loss_vel + loss_acc

    def continuity_loss_non_speech(self, B_c, V_s):
        loss = 0
        for tau in range(-self.tau, self.tau + 1):
            if tau == 0:
                continue
            shifted_B_c = torch.roll(B_c, shifts=tau, dims=2)
            loss += torch.mean(torch.abs(B_c - shifted_B_c) * V_s)
        return loss

    def forward(self, enhancedwav, cleanwav):
        B_c = self.compute_mel_spectrum(enhancedwav)
        B_s = self.compute_mel_spectrum(cleanwav)

        B_vel_c, B_acc_c = self.compute_differential_spectrum(B_c)
        B_vel_s, B_acc_s = self.compute_differential_spectrum(B_s)

        batch_size = cleanwav.shape[0]
        V = []
        for i in range(batch_size):
            clean_np = cleanwav[i].to('cpu').numpy()
            clean_np = clean_np / np.max(np.abs(clean_np))
            clean_np = (clean_np * 32767).astype(np.int16)
            vad = vad_collector(sp=clean_np, fs=16000, frame_size=320, vad=webrtcvad.Vad(3))
            V.append(vad)
        V = np.array(V, dtype=np.float32)
        V = torch.tensor(V, dtype=torch.float32, device=cleanwav.device)
        V = V.unsqueeze(1)

        V_s = 1 - V

        L_env = self.envelope_loss(B_c, B_s, V)
        L_sp = self.continuity_loss_speech(B_vel_c, B_vel_s, B_acc_c, B_acc_s, V)
        L_sil = self.continuity_loss_non_speech(B_c, V_s)

        SISNR_Loss = SISNRLoss()
        denoising_loss = SISNR_Loss(enhancedwav, cleanwav)

        L_NDA = denoising_loss + self.gamma * L_env + self.mu * L_sp + self.omega * L_sil
        return L_NDA


if __name__ == "__main__":
    loss_fn = NDALoss(sample_rate=16000, n_fft=320, hop_length=160, n_mels=60, gamma=5, mu=20, omega=20, eta=3, tau=2)

    enhancedwav = torch.randn(2, 32000)
    cleanwav = torch.randn(2, 32000)

    loss = loss_fn(enhancedwav, cleanwav)
    print("Loss: ", loss.item())
