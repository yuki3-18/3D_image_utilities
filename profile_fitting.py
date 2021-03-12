import numpy as np
import matplotlib.pyplot as plt
from fitting_algorithm import gauss, scipy_fit
import dataIO as io
from utils import MSE

# setting
# center of the vessel
x_center, y_center, z_center = 6, 4, 7
size = 9
# ROI
s_roi = 1
e_roi = size - 1
roi = e_roi - s_roi
# path
path = "./input/"
file = "data"

# input
print("load data")
img = io.read_mhd_and_raw(path + file + ".mhd", 'double')
img = np.reshape(img, (size, size, size))

# roi
profile = img[z_center:z_center + 1, y_center:y_center + 1, s_roi:e_roi]

# profile = np.reshape(profile, (roi, 1))
x_fit = np.linspace(0, roi - 1, roi * 10)

# fitting
print("start fitting")
x = np.linspace(0, roi - 1, roi)
y = np.reshape(profile, (roi))

print("-" * 20, "Gaussian fitting", "-" * 20)
r = scipy_fit(x, y)
print("[A, mu, sigma, o] =", r)
y_pred = gauss(x, r[0], r[1], r[2], r[3])

# calculate error
MSE = MSE(y, y_pred)
print("MSE =", MSE)

# plot
fig = plt.figure()
fig.suptitle('Params: A=%.2f, σ=%.3f, B=%.2f' % (r[0], r[2], r[3]))
plt.scatter(x, y, label='Vessel profile')
plt.plot(x_fit, gauss(x_fit, r[0], r[1], r[2], r[3]), label='Fitting result', color='orange')
plt.ylim([-1000, -250])
plt.xticks(color="None")
plt.ylabel('CT value [H.U.]')
plt.legend(loc='best')
plt.savefig("./output/" + file + ".png")
plt.show()
