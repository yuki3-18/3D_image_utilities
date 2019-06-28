import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab as plb
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import norm
import seaborn as sns
import ioFunction_version_4_3 as IO

#input
label ="L3"
x_center = 12
y_center = 12
z_center = 12
size = 25

# if label == "L1" or "L2" or "L3":
#     size = 25
#     x_center = y_center = z_center = 12
# elif label == "M1" or "M2" or "M3":
#     size = 17
#     x_center = y_center = z_center = 8
# else:
#     size = 9
#     x_center = y_center = z_center = 4


s_roi = 4
e_roi = size - 5
roi = e_roi - s_roi

e = 0.1

path = "E:/from_kubo/vector_rotation/x64/Release/output/"
file = "/output_19_18_10.raw"
    # "/output_9_12_8.raw""/output_13_14_10.raw"

print("load data")
img = IO.read_raw(path+label+file, 'double')
img = np.reshape(img, (size,size,size))
print(img.shape)

# 縦
profile = img[z_center:z_center+1,y_center:y_center+1,s_roi:e_roi]
# 横
# profile = img[z_center:z_center+1,s_roi:e_roi,x_center:x_center+1]

profile = np.reshape(profile,(roi,1))
plt.figure(figsize=(6, 4))
plt.title(r'$\mathrm{Vessel\ \Profile}$')
plt.plot(profile)
plt.savefig(path+label+"/profile.png")

# data fitting
def Caruanas(x,y):
    '''
    Caruanas algorithm
    (x,y)の観測値からその正規分布のパラメータを推定
    普通の最小二乗法
    '''

    A = np.zeros((3,3))
    A[0][0] = float(len(x))
    sum_x = np.sum(x)
    A[0][1] = sum_x
    A[1][0] = sum_x
    sum_x2 = np.sum(x**2)
    A[0][2] = sum_x2
    A[1][1] = sum_x2
    A[2][0] = sum_x2
    sum_x3 = np.sum(x**3)
    A[1][2] = sum_x3
    A[2][1] = sum_x3
    A[2][2] = np.sum(x**4)

    b = np.zeros(3)
    b[0] = np.sum(np.log(y))
    b[1] = np.sum(x*np.log(y))
    b[2] = np.sum(x**2 * np.log(y))

    ans = np.linalg.solve(A,b)

    mu = -ans[1]/(2.0*ans[2])
    sigma = np.sqrt(-0.5/(ans[2]))
    A_factor = np.exp(ans[0]-ans[1]**2*0.25/(ans[2]))
    return mu,sigma,A_factor

def Guos(x,y):
    '''
    GUOS algorithm
    誤差について1次の項まで考えてる
    '''

    A = np.zeros((3,3))
    b = np.zeros(3)

    A[0][0] = np.sum(y**2)
    xy_2 = np.sum(x*y**2)
    A[0][1] = xy_2
    A[1][0] = xy_2
    x_2y_2 = np.sum((x**2)*(y**2))
    A[0][2] = x_2y_2
    A[1][1] = x_2y_2
    A[2][0] = x_2y_2
    x_3y_2 = np.sum((x**3)*(y**2))
    A[1][2] = x_3y_2
    A[2][1] = x_3y_2
    x_4y_2 = np.sum((x**4)*(y**2))
    A[2][2] = x_4y_2

    b[0] = np.sum((y**2)* np.log(y))
    b[1] = np.sum(x*(y**2)*np.log(y))
    b[2] = np.sum((x**2)*(y**2)*np.log(y))

    ans = np.linalg.solve(A,b)

    mu = -ans[1]/(2.0*ans[2])
    sigma = np.sqrt(-0.5/(ans[2]))
    A_factor = np.exp(ans[0]-ans[1]**2*0.25/(ans[2]))
    return mu,sigma,A_factor

def scipy_fit(x_in,y_in):
    '''
    scipyのcurve_fit使う.
    '''
    def gaussian_fit(x,A,mu,sigma):
        return A*np.exp(-(x-mu)**2 / (2.0*sigma**2))

    params,cov = curve_fit(gaussian_fit,x_in,y_in)
    # params = [A,mu,sigma]なので上の二つと順番が違う
    return params


offset = abs(min(profile)) + e

print("start fitting")
x = np.linspace(0,roi-1,roi)
y = np.reshape(profile,(roi)) + offset

print("x =",x)
print("y =",y)

print("scipy_fit")
p = scipy_fit(x,y)
print("A,mu,sigma =",p)

print("Caruanas")
r = Caruanas(x,y)
print("mu,sigma,A =",r)

print("Guos")
m = Guos(x,y)
print("mu,sigma,A =",m)

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))

x_fit = np.linspace(0,roi-1,roi*10)

# p
plt.figure(figsize=(6, 4))
plt.scatter(x, y, label='Data')
plt.plot(x_fit, gauss(x_fit,p[0], p[1], p[2]), label='Gaussian', color='orange')
plt.title('Parameters: C=%.2f, sigma=%.5f' %(p[0], abs(p[2])))
plt.legend(loc='best')
plt.savefig(path+label+"/scipy.png")
plt.show()

# r
plt.figure(figsize=(6, 4))
plt.scatter(x, y, label='Data')
plt.plot(x_fit, gauss(x_fit,r[2], r[0], r[1]), label='Gaussian', color='orange')
plt.title('Parameters: C=%.2f, sigma=%.5f' %(r[2], abs(r[1])))
plt.legend(loc='best')
plt.savefig(path+label+"/Caruanas.png")
plt.show()

# m
plt.figure(figsize=(6, 4))
plt.scatter(x, y, label='Data')
plt.plot(x_fit, gauss(x_fit,m[2], m[0], m[1]), label='Gaussian', color='orange')
plt.title('Parameters: C=%.2f, sigma=%.5f' %(m[2], abs(m[1])))
plt.legend(loc='best')
plt.savefig(path+label+"/Guos.png")
plt.show()