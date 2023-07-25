import librosa
import librosa.display
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
y1, sr = librosa.load('huatu/6227/6227-60173-0001.wav', sr=16000,offset=0, duration=None)
y2, sr = librosa.load('huatu/fgsm6227/6227-60173-0001.wav', sr=16000,offset=0, duration=None)

y3, sr = librosa.load('huatu/pgd6227/6227-60173-0001.wav', sr=16000,offset=0, duration=None)
y4, sr = librosa.load('huatu/mim6227/6227-60173-0001.wav', sr=16000,offset=0, duration=None)
y5, sr = librosa.load('huatu/cw6227/6227-60173-0001.wav', sr=16000,offset=0, duration=None)
y=y5-y1

print(y)
print(sr)
print(len(y))

plt.figure()
librosa.display.waveplot(y, sr)
plt.title('Waveform', fontproperties="SimSun")
#plt.xlim(left = 0,right = 6)
plt.ylim(bottom=-0.008,top=0.008)
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()




melspec = librosa.feature.melspectrogram(y, sr)
logmelspec = librosa.power_to_db(melspec)


fig, ax = plt.subplots(1,1)
img = librosa.display.specshow(logmelspec, y_axis='mel', sr=sr, x_axis='time')

plt.title('spectrogram', fontproperties="SimSun")
fig.colorbar(img, ax=ax, format="%+2.f dB")

plt.ylim(0, 8192)
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()