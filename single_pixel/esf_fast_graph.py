import numpy as np
import matplotlib.pyplot as plt

file = r'D:\OneDrive - University of Victoria\Research\Single Pixel\ESF Checks\fast scan 5micron s.txt10.000000.txt'

data = open(file).read().split()
times = np.array(data[1::2], dtype='float') / 1000
data = np.array(data[2::2], dtype='float')

fig = plt.figure(figsize=(8, 6))
plt.plot(times[42:1500], data[42:1500])
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Signal (nA)', fontsize=14)
plt.tick_params(labelsize=12)
plt.title(r'Fast-scan (moving at 5 $\mu$m/s)', fontsize=15)
plt.show()


averaged = []
pix = 0
avg = 0


for i, time in enumerate(times[0:-1]):
    curr_time = times[i+1] - times[i]
    if pix + curr_time < 4:
        pix = pix + curr_time
        avg = avg + data[i]
    else:
        temp = 4 - pix
        per = temp / curr_time
        avg = avg + per * data[i]

        averaged.append(avg)
        pix = curr_time - temp
        avg = (1 - per) * data[i]

averaged = np.array(averaged)

# plt.plot(averaged)
# plt.show()
