import numpy as np
import matplotlib.pyplot as plt

def ft(samples, Fs, t0):
    """Approximate the Fourier Transform of a time-limited signal 
    by means of the discrete Fourier Transform.
    
    samples: signal values sampled at the positions t0 + n/Fs
    Fs: Sampling frequency of the signal
    t0: starting time of the sampling of the signal
    """
    f = np.linspace(-Fs/2, Fs/2, len(samples), endpoint=False)
    return np.fft.fftshift(np.fft.fft(samples)/Fs * np.exp(-2j*np.pi*f*t0))


Fs = 1000 #sampling frequency
t0 = 2 #half time interval we look at
t = np.arange(-t0, t0, 1/Fs) #time samples

g = lambda t: np.cos(2*np.pi*t)+np.sin(3*np.pi*t)+1j*np.cos(5*np.pi*t)+1j*np.sin(7*np.pi*t)

plt.subplot(3,3,1)
plt.plot(t, g(t).real, label="Real part")
plt.plot(t, g(t).imag, label="Imaginary part")
plt.plot(t, abs(g(t)), label="Absolute value")
plt.title("Time domain signal x(t)")
plt.legend()

f = np.arange(-Fs/2, Fs/2, Fs/len(t)) #frequency samples
ft_g = np.round(ft(g(t), Fs, -t0)/4, decimals=1)

plt.subplot(3,3,2)
plt.title("Fourier transform of real part of x(t)")
ft_real_g = np.round(ft(g(t).real, Fs, -t0)/4,decimals=1)
plt.stem(f, ft_real_g.real, linefmt='b-', markerfmt='bo', basefmt='b', label='Real')
plt.stem(f, ft_real_g.imag, linefmt='r-', markerfmt='ro', basefmt='r', label='Imag')
plt.xlim(-10,10)
plt.legend()


flipped = np.conjugate(np.flip(ft_g))
flipped = np.insert(flipped, 0, 0)
flipped = flipped[:-1]
newsignal=0.5*(ft_g+flipped)
plt.subplot(3,3,3)
plt.title("Even part (conjugate symmetric) of X(w)")
plt.stem(f, newsignal.real, linefmt='b-', markerfmt='bo', basefmt='b', label='Real')
plt.stem(f, newsignal.imag, linefmt='r-', markerfmt='ro', basefmt='r', label='Imag')
plt.xlim(-10,10)
plt.legend()


plt.subplot(3,3,4)
plt.title("Fourier transform of imaginary part of x(t)")
ft_real_g1 = np.round(ft(1j*g(t).imag, Fs, -t0)/4,decimals=1)
plt.stem(f, ft_real_g1.real, linefmt='b-', markerfmt='bo', basefmt='b', label='Real')
plt.stem(f, ft_real_g1.imag, linefmt='r-', markerfmt='ro', basefmt='r', label='Imag')
plt.xlim(-10,10)
plt.legend()

ft_g = np.round(ft(g(t), Fs, -t0)/4, decimals=1)
flipped = np.conjugate(np.flip(ft_g))
flipped = np.insert(flipped, 0, 0)
flipped = flipped[:-1]
newsignal=0.5*(ft_g-flipped)


plt.subplot(3,3,5)
plt.title("Odd(conjugate skew symmetric) part of X(w)")
plt.stem(f, newsignal.real, linefmt='b-', markerfmt='bo', basefmt='b', label='Real')
plt.stem(f, newsignal.imag, linefmt='r-', markerfmt='ro', basefmt='r', label='Imag')
plt.xlim(-10,10)
plt.legend()

flipped=np.conjugate(np.flip(g(t)))
flipped=np.insert(flipped,0,0)
flipped=flipped[:-1]
even_x_t=0.5*(g(t)+flipped)
ft_e_x = np.round(ft(even_x_t, Fs, -t0)/4, decimals=1)

plt.subplot(3,3,6)
plt.title("Fourier transform of even part of x(t)")
plt.stem(f, ft_e_x.real, linefmt='b-', markerfmt='bo', basefmt='b', label='Real')
plt.stem(f, ft_e_x.imag, linefmt='r-', markerfmt='ro', basefmt='r', label='Imag')
plt.xlim(-10,10)
plt.legend()

plt.subplot(3,3,7)
plt.title("Real part of X(w)")
plt.stem(f, ft_g.real, linefmt='b-', markerfmt='bo', basefmt='b', label='Real')
plt.xlim(-10,10)
plt.legend()

flipped=np.conjugate(np.flip(g(t)))
flipped=np.insert(flipped,0,0)
flipped=flipped[:-1]
odd_x_t=0.5*(g(t)-flipped)
ft_o_x = np.round(ft(odd_x_t, Fs, -t0)/4, decimals=1)

plt.subplot(3,3,8)
plt.title("Fourier transform of odd part of x(t)")
plt.stem(f, ft_o_x.real, linefmt='b-', markerfmt='bo', basefmt='b', label='Real')
plt.stem(f, ft_o_x.imag, linefmt='r-', markerfmt='ro', basefmt='r', label='Imag')
plt.xlim(-10,10)
plt.legend()

plt.subplot(3,3,9)
plt.title("Imaginary part of X(w)")
plt.stem(f, ft_g.imag)
plt.xlim(-10,10)




plt.show()
