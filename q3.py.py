import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

speech, fs = sf.read(r'D:\IIT MANDI\Sem 1\CS571 Programming Practicum\LAB 9\Ques3\should.wav')
t = np.array(np.linspace(0, len(speech), num = len(speech))) / fs

print("Sampling rate of signal is {0} Hz".format(fs))

plt.figure(1)
plt.plot(t, speech)
plt.title("Speech")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

def enframe(x,winsize,hoplength,fs,wintype = 'rect'):
    frames = []
    
    winSam = int(fs*winsize)    # No. of samples in window
    hopSam = int(fs*hoplength)  # Frame shift size
    
    numOfFrames = int( (len(x) - winSam) / hopSam) + 1  

    window = np.ones(winSam - hopSam)   
    if wintype == 'hamm':
        window = np.hamming(winSam - hopSam)
    
    step = winSam - hopSam
    for i in range(numOfFrames):
        startIndx = i*step
        endIndx = (i+1)*step
        frames.append( x[startIndx:endIndx]*window )     # Appending frames to the frames
    return frames


winsize = 30/1000    # winsize is 30 ms
hoplength = 15/1000  # hoplength is 15 ms 

## Frames from Speech Signal  
wintype = 'hamm'  ##rectangular
frames = enframe(speech,winsize,hoplength,fs, wintype)    

## Spectrum of one frame
def frame_spectrum(frames, frame_no):
    frame = frames[frame_no]
    n = np.arange(len(frame))
    
    # Getting DFT spectrum
    dft_frame = np.fft.fft(frame)
    
    freq = n/len(n)
    logmag_dft_frame = np.log(np.abs(dft_frame))
   
    fig, p = plt.subplots(nrows = 2, figsize=(10,8))
    fig.tight_layout()
    
    # First subplot
    p[0].plot(n, frame, color = 'green')
    p[0].set_label("Frame")
    p[0].set_xlabel("Samples")
    p[0].set_ylabel("Amplitude")
    p[0].set_title(str(frame_no) + " Frame  + " + wintype + "  Window")
    
    # Second subplot
    p[1].plot(freq, logmag_dft_frame, color = 'blue')
    p[1].set_label("Log Magnitude")
    p[1].set_xlabel("Freq in π")
    p[1].set_ylabel("log[xn]")
    p[1].set_title("DFT spectrum of frame in log scale")
    plt.show()
    
frame_spectrum(frames, 25)

from scipy import signal 

## Spectrogram using Hamming Window
f, t, Zxx = signal.stft(speech, fs, window='hamm', nperseg=150)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude using Hamming window')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.show()

## Spectrogram using Rectangular Window
f, t, zx = signal.stft(speech, fs, window='rect', nperseg=150)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude using Rectangular window')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.show()

