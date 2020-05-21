import numpy as np
import matplotlib.pyplot as plt
import ioFunction_version_4_3 as IO

def SNR(data):
    avr = np.average(data)
    std = np.std(data)
    SNR = avr/std
    print("SNR={}, avr={}, std={}".format(abs(SNR), avr, std))
    print(10*np.log10(SNR**2), "[dB]")

img = IO.read_raw("E:/data/bkg/bkg2.raw", 'short')
img = np.reshape(img, (-1,1))
img = np.reshape(img,(9,9,9))
img = img[2:7,2:7,2:7]
img = np.reshape(img,(125,))

SNR(img)

plt.hist(img,bins=60)
plt.show()
plt.plot(img)
plt.show()