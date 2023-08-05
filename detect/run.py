import numpy as np
import pyaudio
import wave
import librosa
import torch
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from GhoDenNet import *


device = torch.device("cpu")
model = GhoDenNet()
model = torch.load('GhoDenNet.pth', map_location='cpu')
model = model.to(device)
model.eval()

print('detecting：')

var = 1
while var == 1:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    fs = 16000
    duration = 3.1
    channels = 1
    n = duration * fs
    t = np.arange(1, n) / fs
    wave_output_file = 'record.wav'

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=channels, rate=fs,
                    input=True, frames_per_buffer=CHUNK)


    frames = []
    for i in range(0, int(fs / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wave_output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    wav, sr = librosa.load('./record.wav', sr=16000)
    spec_mag = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=512, hop_length=256, n_mels=128)
    logmelspec = librosa.amplitude_to_db(spec_mag)
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_fft=512, hop_length=256, n_mfcc=128)
    channels = np.stack([spec_mag, logmelspec, mfcc], axis=0)
    channels = channels.astype('float32')
    channels = channels[:, :, :192]
    mean = np.mean(channels, axis=1, keepdims=True)
    std = np.std(channels, axis=1, keepdims=True)
    channels = (channels - mean) / (std + 1e-5)
    channels = torch.tensor(channels)
    channels = channels.unsqueeze(0)

    out = model(channels)
    out = out.data.cpu().numpy()
    out = out[0]
    a,b,c = out[0],out[1],out[2]
    print('·')
    if a > 4:
        print('class：cough',datetime.now())
    elif b > 4.5:
        print('class：siren',datetime.now())
    else:
        pass


