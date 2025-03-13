import numpy as np
from scipy.stats import bernoulli as Bern


def t_gen(x,W,n):
    logit = np.matmul(W,x) - n
#     print(logit)
    sigmoid = 1/(1+np.exp(-logit))
#     print(sigmoid)
    t = Bern.rvs(sigmoid)
#     print(t)
    return t

def y_gen(x,W,n,wty):
    C = 1.5
    ty = wty * np.array([-1, 1])
    y = C * (np.matmul(W,x) + ty + n)
    return y[0],y[1]

def z_gen(mu, sigma):
    

def _simulate(data_in, mode: str='train'):
    # covariates = normalize(data_in['x'][:,:,0])
    covariates = data_in['x'][:,:,0]
    n_in = len(covariates)

    x_mat = []
    t_mat = []
    f_mat = []
    cf_mat = []
    mu0_mat = []
    mu1_mat = []

    for i in range(10):
        t_list = []
        f_list = []
        cf_list = []
        mu0_list = []
        mu1_list = []

        diagy = np.zeros((25,25),int)
        np.fill_diagonal(diagy,1)
        diagy = 0.1*diagy
        Wy = np.random.multivariate_normal(np.zeros(25),diagy,2)

        diagt = np.zeros((25,25),int)
        np.fill_diagonal(diagt,1)
        diagt = 0.1*diagt
        Wt = np.random.multivariate_normal(np.zeros(25),diagt)

        Wty = np.random.normal(0, 1)
        
        for x in covariates:
            #print(x.shape)
            nt = np.random.normal(2, 0.01)
            t = t_gen(x,Wt,nt)
            t_list.append(t)
            
            ny = np.random.normal(0, 0.01, (2,))
            y0, y1 = y_gen(x,Wy,ny,Wty)
            if t == 0:
                f_list.append(y0)
                cf_list.append(y1)
            else:
                f_list.append(y1)
                cf_list.append(y0)
            
            mu0_list.append(y0)
            mu1_list.append(y1)
                
        print(sum(t_list) / n_in)
        x_mat.append(covariates.tolist())
        t_mat.append(t_list)
        f_mat.append(f_list)
        cf_mat.append(cf_list)
        mu0_mat.append(mu0_list)
        mu1_mat.append(mu1_list)

    XS = np.swapaxes(np.swapaxes(x_mat,0,2),0,1)
    print('XS: ', XS.shape)

    T = np.swapaxes(t_mat,0,1)
    print('T: ', T.shape)

    F = np.swapaxes(f_mat,0,1)
    print('F: ', F.shape)

    CF = np.swapaxes(cf_mat,0,1)
    print('CF: ', CF.shape)

    MU0 = np.swapaxes(mu0_mat,0,1)
    print('MU0: ', MU0.shape)

    MU1 = np.swapaxes(mu1_mat,0,1)
    print('MU1: ', MU1.shape)

    return XS, T, F, CF, MU0, MU1