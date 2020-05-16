import numpy as np
from scipy.optimize import curve_fit

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
    return mu, sigma, A_factor, ans[0], ans[1], ans[2]

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

    mu = -ans[1] / (2.0 * ans[2])
    sigma = np.sqrt(-0.5 / (ans[2]))
    A_factor = np.exp(ans[0]-ans[1]**2*0.25/(ans[2]))
    return mu, sigma, A_factor, ans[0], ans[1], ans[2]

def scipy_fit(x_in, y_in):
    '''
    scipy curve_fit
    '''
    param_initial = np.array([667, 3, 0.9, -667]) # initial guess
    param_bounds = ((0, 2, 0.5, -1300), (1300, 5, 2., 0))  # bounds for parameter (A, mu, sigma, o)
    params, cov = curve_fit(gauss, x_in, y_in, p0=param_initial, bounds=param_bounds)
    #params = [A,mu,sigma]なので上の二つと順番が違う
    return params

def gauss(x, A, mu, sigma, o):
    return A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2)) + o