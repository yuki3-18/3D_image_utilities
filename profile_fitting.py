import numpy as np
import matplotlib.pyplot as plt
from fitting_algorithm import gauss, scipy_fit
import ioFunction_version_4_3 as IO
from utils import MSE

# setting
x_center, y_center, z_center = 4, 4, 4
size = 9

s_roi = 1
e_roi = size - 1
roi = e_roi - s_roi

path = "E:/from_kubo/vector_rotation/x64/Release/output/"
file = "output_5_4_6"
    # "output_7_6_4""output_3_7_2"

# input
print("load data")
img = IO.read_mhd_and_raw(path + file + ".mhd", 'double')
img = np.reshape(img, (size, size, size))

# row
profile = img[z_center:z_center + 1, y_center:y_center + 1, s_roi:e_roi]

# profile = np.reshape(profile, (roi, 1))
x_fit = np.linspace(0, roi - 1, roi * 10)

# fitting
print("start fitting")
x = np.linspace(0, roi - 1, roi)
y = np.reshape(profile, (roi))

print("-"*20, "Gaussian fitting", "-"*20)
r = scipy_fit(x, y)
print("[A, mu, sigma, o] =", r)
y_pred = gauss(x, r[0], r[1], r[2], r[3])

# calculate error
MSE = MSE(y, y_pred)
print("MSE =", MSE)

# plot
fig = plt.figure()
fig.suptitle('Params: A=%.2f, σ=%.3f, C=%.2f' %(r[0], r[2], r[3]))
plt.scatter(x, y, label='Vessel profile')
plt.plot(x_fit, gauss(x_fit, r[0], r[1], r[2], r[3]), label='Fitting result', color='orange')
plt.ylim([-1000, -250])
plt.xticks(color="None")
plt.ylabel('CT value [H.U.]')
plt.legend(loc='best')
plt.savefig("E:/GoogleDrive/master/results/analise/" + file + ".png")
plt.show()