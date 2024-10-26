# -----------------------------------------------------
# Title: Simple Command Line Audio Equalizer
# Libraries required: librosa, numpy, scipy, matplotlib, pydub 
# Description: Takes in an audio file with given duraton and changes the frequency responses of specific bands
# Author: EE23B002 ABHINAV I S
# Date: March 26, 2024
# Version: 2.0
# -----------------------------------------------------

import librosa
import numpy as np
from scipy.fft import fft, ifft
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play

#head
print("Simple CLI Audio Eqlaizer")

#loading the file
def select_audio():
    filename = input("Name of file\n > ")
    duration = int(input("Enter duration to be imported"))
    audio_data, fs = librosa.load(filename, duration=duration)
    return audio_data, fs

#actually changing the freq response
def equalize(low_limit, high_limit, val, fft_data):
    L=int(len(fft_data))
    #if band goes higher than the Nyquist freq
    if high_limit>fs/2:
        high_limit=int(fs/2)
    if low_limit>fs/2:
        low_limit=int(fs/2)
    #changing frequency response from beginning and ending(as required) as it is repeated
    for i in range(L-int(high_limit*L/fs),L-int(low_limit*L/fs)):
        fft_data[i]*=val
    for i in range(int(low_limit*L/fs),int(high_limit*L/fs)):
        fft_data[i]*=val
    


#selection of mode
while(1):
    print("Mode ")
    print("1: Select audio file")
    print("2: Equalize")
    print("3: Play final sound")
    print("4: Plot Graphs")
    selection = int(input(""))
    if selection == 1:
        audio_data, fs = select_audio()
    elif selection == 2:
        fRes_0_300 = int(input("Frequency Response for 0-300Hz: "))
        fRes_300_1K = int(input("Frequency Response for 300-1KHz: "))
        fRes_1K_3K = int(input("Frequency Response for 1Khz-3KHz: "))
        fRes_3K_8K = int(input("Frequency Response for 3K-8KHz: "))
        fRes_8K_14K = int(input("Frequency Response for 8K-14KHz: "))
        fRes_14K_20K = int(input("Frequency Response for 14K-20KHz: "))
        fft_data = fft(audio_data)
        equalize(0,300,fRes_0_300, fft_data)
        equalize(300,1000,fRes_300_1K, fft_data)
        equalize(1000,3000,fRes_1K_3K, fft_data)
        equalize(3000,8000,fRes_3K_8K, fft_data)
        equalize(8000,14000,fRes_8K_14K, fft_data)
        equalize(14000,20000,fRes_14K_20K, fft_data)
    elif selection==3:
        time_domain_signal = np.real(ifft(fft_data))
        write("out.wav", fs, time_domain_signal)
        audio_file_path = 'out.wav'  
        audio = AudioSegment.from_file(audio_file_path)

        # Play the audio
        play(audio)
        plt.show()

    elif selection == 4:
        plt.subplot(5, 1, 1)
        plt.title("Time domain Audio")
        time_axis = np.linspace(0, 10 ,num=len(audio_data),endpoint=True)
        plt.plot(time_axis, audio_data)
        

        fft_before=fft(audio_data)
        plt.subplot(5, 1, 3)
        plt.title("Frequency Domain before equalization")
        freq = np.fft.fftfreq(len(fft_data), 1/fs)
        plt.plot(freq, np.abs(np.fft.fftshift(fft_before)))
        
        plt.subplot(5,1,5)
        plt.title("Frequency Domain after equalization")
        plt.plot(freq, np.abs(np.fft.fftshift(fft_data)))
        plt.show()
        # Inverse Fourier Transform
    else:
        break




