from torch.utils.data import Dataset
import torchaudio
import torch
import numpy as np
from scipy.interpolate import interp1d

# def load_audio(audio_path, sample_rate):
#     audio_data = torchaudio.load(audio_path, sr=sample_rate)
#     audio_data = torch.FloatTensor(audio_data)

#     return audio_data

class AudioDataset(Dataset):
    def __init__(self, data, _id, root, audio_conf, frame_length, shift_length, train=True):
        
        self.data = data
        self.root = root
        self.train = train
        self._id = _id

        # if val:
        #     self.data = data
        # else:
        #     self.data = [sample for sample in data if _id in sample]

        if not self.train:
            anomaly_data = [data for data in self.data if "anomaly" in data or self._id not in data]
            self.anomaly_num = len(anomaly_data)
            self.normal_num = len(self.data) - self.anomaly_num
            #print(len(self.data))

        self.frame_length = frame_length
        self.shift_length = shift_length

        self.audio_conf = audio_conf
        """
        Audio Config is a dict type
            num_mel_bins (128):     number of mel bins in audio spectrogram
            target_length (1024):   number of frames is formed after the raw audio go through the filter bank, use 1024
            freqm (0):              frequency mask length, default: 0
            timem (0):              time mask length, default: 0
            mixup (0):              mixup with other wave
            dataset (dcase):        the dataset we apply on is dcase dataset
            mode (train or test):   train mode or test mode
            mean (-4.2677393):      use for normalization
            std (4.5689974):        use for normalization
            noise (False):          does the dataset add noise into audio
        """
        
        self.melbins = audio_conf['num_mel_bins']
        self.target_length = audio_conf['target_length']
        self.freqm = audio_conf['freqm']
        self.timem = audio_conf['timem']
        self.mixup = audio_conf['mixup']
        self.dataset = audio_conf['dataset']
        self.mode = audio_conf['mode']
        self.norm_mean = audio_conf['mean']
        self.norm_std = audio_conf['std']
        self.noise = audio_conf['noise']
        self.sample_rate = audio_conf['sample_rate']
    
    def _wav2fbank(self, filename, filename2=None):
        # no mixup
        if filename2 == None:

            # transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            # transform2 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate*2)
            # transform3 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate*4)
            waveform, sr = torchaudio.load(filename)
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
            waveform2 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate*2)(waveform)
            waveform3 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate*4)(waveform)

            waveform = waveform - waveform.mean()
            waveform2 = waveform2 - waveform2.mean()
            waveform3 = waveform3 - waveform3.mean()
            # print("waveform shape", waveform.shape, waveform2.shape, waveform3.shape)
        """
        sample rate: 16k
        frame length: 50ms (default)
        shift length: 20ms
        """
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.sample_rate, use_energy=False, 
                                                  frame_length=self.frame_length, window_type='hanning', 
                                                  num_mel_bins=self.melbins, dither=0.0, frame_shift=self.shift_length)
        fbank2 = torchaudio.compliance.kaldi.fbank(waveform2, htk_compat=True, sample_frequency=self.sample_rate*2, use_energy=False, 
                                                   frame_length=self.frame_length, window_type='hanning', 
                                                   num_mel_bins=self.melbins*2, dither=0.0, frame_shift=self.shift_length)
        fbank3 = torchaudio.compliance.kaldi.fbank(waveform3, htk_compat=True, sample_frequency=self.sample_rate*4, use_energy=False, 
                                                   frame_length=self.frame_length, window_type='hanning', 
                                                   num_mel_bins=self.melbins*4, dither=0.0, frame_shift=self.shift_length)
        # print("fbank shape:", fbank.shape, fbank2.shape, fbank3.shape)
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
            fbank2 = m(fbank2)
            fbank3 = m(fbank3)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]
            fbank2 = fbank2[0:self.target_length, :]
            fbank3 = fbank3[0:self.target_length, :]
        if filename2 == None:
            return [fbank, fbank2, fbank3], 0
        """
        else:
            return fbank, mix_lambda
        """

    def __getitem__(self, index):
        file_path = self.data[index]
        # no mixup
        fbanks, _ = self._wav2fbank(file_path)
        fbank, fbank2, fbank3 = fbanks

        # Normalize the audio data
        fbank = (fbank - self.norm_mean) / (self.norm_std*2)
        fbank2 = (fbank2 - self.norm_mean) / (self.norm_std*2)
        fbank3 = (fbank3 - self.norm_mean) / (self.norm_std*2)


        if self.train:
            return [fbank, fbank2, fbank3]
        else:
            return [fbank, fbank2, fbank3], 1 if "anomaly" in file_path or self._id not in file_path else 0
    
    def __len__(self):
        return len(self.data)