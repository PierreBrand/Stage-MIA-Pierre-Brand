
# Version 11

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Bibliothèques
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import numpy.linalg as nla
import pandas as pnd
import sklearn.covariance as sklcov
import sklearn.cluster as sklclu
import sklearn.metrics.cluster as sklmc
import itertools as itt
import scipy.cluster.hierarchy as spch

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions peut-être utiles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def norme(x):
    return np.sqrt(x@x)

def angle_oriente(x):
    nx = x/norme(x)
    if nx[1] >= 0:
        return np.arccos(nx[0])
    else :
        return -np.arccos(nx[0])

def rotation(a):
    R = np.array([[np.cos(a), np.sin(a)],[-np.sin(a), np.cos(a)]])
    return R

def erreur_Z(Z1,Z2,a,sym=False):
    S = np.array([[1,0],[0,1-2*int(sym)]])
    R = np.transpose(rotation(a)@S)
    RZ2 = Z2@R
    return np.sum((Z1-RZ2)**2)

def orthogonalize(W):
    R2 = np.transpose(W)@W
    SpR2, P = nla.eig(R2)
    R = np.real(P @ np.diag(1/np.sqrt(SpR2)) @ nla.inv(P))
    W_2 = W @ R
    return W_2

def normalize(P):
    P_norms = np.sqrt(np.sum(P**2,axis=0))
    D_P = np.diag(1/P_norms)
    P2 = P@D_P
    return P2

def trace(A):
    N1,N2 = np.shape(A)
    if N1 != N2 :
        print('Erreur de dimensions sur A')
    else :
        return np.sum([A[n][n] for n in range(N1)])

def cov_emp(Y):
    N,D = np.shape(Y)
    mu_Y = np.mean(Y,axis=0)
    Yc = Y - mu_Y
    S = 1/N * np.transpose(Yc) @ Yc
    return S

def L_knee(x,alpha=1.0):
    
    N = len(x)
    ordre = np.sort(x)
    
    #Renormalisation
    vert_length = ordre[-1]-ordre[0]
    x1_list = np.linspace(0,1,N)
    y1_list = np.array([(y-ordre[0])/vert_length for y in ordre])
    z1_list = alpha*y1_list - x1_list
    
    n_star = int(np.argmax(z1_list))
    return n_star

def R_elbow(x,alpha=1.0):
    
    N = len(x)
    ordre = np.sort(x)
    
    #Renormalisation
    vert_length = ordre[-1]-ordre[0]
    x1_list = np.linspace(0,1,N)
    y1_list = np.array([(y-ordre[0])/vert_length for y in ordre])
    z1_list = x1_list - alpha*y1_list
    
    n_star = int(np.argmax(z1_list))
    return n_star

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions pour simuler les données
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sim_param(D,L,m1=0.0,m2=0.0,m3=0.0,s1=1.0,s2=1.0,s3=1.0,disp=False,orthog=True):
    W = rd.normal(m1,s1**2,(D,L))
    mu = rd.normal(m2,s2**2,D)
    sigma2 = rd.normal(m3,s3**2)**2
    
    if orthog:
        W = orthogonalize(W)
    
    if disp:
        print('$W = $', W)
        print('$\mu = $', mu)
        print('$\sigma^2 = $', sigma2)
        
    return W, mu, sigma2

def sim_data(W,mu,sigma2,N,disp=False):
    D,L = np.shape(W)
    Z = rd.normal(0,1,(N,L))
    Y = np.array([W@z + mu + rd.normal(0,sigma2,D) for z in Z])
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
    return Z, Y

def sim_param_cov(D,L,C,m1=0.0,m2=0.0,m3=0.0,m4=0.0,s1=1.0,s2=1.0,s3=1.0,s4=1.0,disp=False,orthog=True):
    W = rd.normal(m1,s1**2,(D,L))
    V = rd.normal(m2,s2**2,(D,C))
    mu = rd.normal(m3,s3**2,D)
    sigma2 = rd.normal(m4,s4**2)**2
    
    if orthog:
        W = orthogonalize(W)
        V = orthogonalize(V)
    
    if disp:
        print('$W = $', W)
        print('$V = $', V)
        print('$\mu = $', mu)
        print('$\sigma^2 = $', sigma2)
        
    return W, V, mu, sigma2

def sim_data_cov(W,V,mu,sigma2,N,Sigma_X=None,disp=False):
    D1,L = np.shape(W)
    D2,C = np.shape(V)
    if D1 != D2 :
        print('Erreur de dimension sur W et V')
    else :
        D = D1
        
    Z = rd.normal(0,1,(N,L))
    E = np.ones((N,1)) @ np.array([mu]) + rd.normal(0,sigma2,(N,D))
    
    if type(Sigma_X) == type(None) :
        X = rd.normal(0,1,(N,C))
    else :
        X = rd.multivariate_normal(np.zeros(C),Sigma_X,N)
    
    Y = Z@np.transpose(W) + X@np.transpose(V) + E
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$X = $', X)
    return Z, Y, X

def sim_Lambda(D,la=1.0,p=0.1):
    pos_def = False
    while not pos_def:
        P = rd.exponential(np.sqrt(la),(D,D))
        signes = 2*rd.choice(2,(D,D)) - 1
        Q = P * signes
        full_LA = np.transpose(Q) @ Q
        
        q = np.sqrt(1-p)
        occur1 = rd.choice(2,(D,D),p=[q,1-q])
        occur2 = ((occur1 + np.transpose(occur1) + np.eye(D)).astype(bool)).astype(float)
        Lambda = occur2*full_LA
        
        SpL,P = nla.eig(Lambda)
        pos_def = np.all(SpL>0)
    return Lambda

def sim_param_LRPSI(D,L,m1=0.0,m2=0.0,s1=0.0,s2=0.0,la=1.0,p=0.1,disp=False,orthog=True):
    W = rd.normal(m1,s1**2,(D,L))
    sigma2 = rd.normal(m2,s2**2)**2
    Lambda = sim_Lambda(D,la,p)
    return W, Lambda, sigma2

def sim_data_LRPSI(W,Lambda,N,sigma2,disp=False):
    D,L = np.shape(W)
    D1,D2 = np.shape(Lambda)
    if D1 != D2 or D1 != D :
        print('Erreur de dimension sur W et Lambda')
        
    Z = rd.normal(0,1,(N,L))
    E = rd.normal(0,sigma2,(N,D))
    X = rd.multivariate_normal(np.zeros(D),nla.inv(Lambda),N)
    Y = Z@np.transpose(W) + X + E
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$X = $', X)
        
    return Z, Y, X

def sim_param_obsmix_1(K,D,L,m1=0.0,m2=0.0,m3=0.0,s1=1.0,s2=1.0,s3=1.0,disp=False,orthog=True):
    thetas = [sim_param(D,L,m1,m2,m3,s1,s2,s3,disp,orthog) for k in range(K)]
    return thetas

def sim_param_obsmix_2(K,D,L,C,m1=0.0,m2=0.0,m3=0.0,m4=0.0,s1=1.0,s2=1.0,s3=1.0,s4=1.0,disp=False,orthog=True):
    thetas = [sim_param_cov(D,L,C,m1,m2,m3,m4,s1,s2,s3,s4,disp,orthog) for k in range(K)]
    return thetas

def sim_param_obsmix_3(K,D,L,m1=0.0,m2=0.0,s1=0.0,s2=0.0,la=1.0,p=0.1,disp=False,orthog=True):
    thetas = [sim_param_LRPSI(D,L,m1,m2,s1,s2,la,p,disp,orthog) for k in range(K)]
    return thetas

def sim_param_latmix_1(K,D,L,m2=0.0,m3=0.0,s2=1.0,s3=1.0,m_glob1=0.0,m_glob2=0.0,m_glob3=0.0,s_glob1=1.0,s_glob2=1.0,s_glob3=1.0,disp=False,orthog=True):
    eta = sim_param(D,L,m_glob1,m_glob2,m_glob3,s_glob1,s_glob2,s_glob3,disp,orthog)
    thetas = [sim_param(L,L,0.0,m2,m3,1.0,s2,s3,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def sim_param_latmix_2(K,D,L,C,m2=0.0,m3=0.0,s2=1.0,s3=1.0,m_glob1=0.0,m_glob2=0.0,m_glob3=0.0,m_glob4=0.0,s_glob1=1.0,s_glob2=1.0,s_glob3=1.0,s_glob4=1.0,disp=False,orthog=True):
    eta = sim_param_cov(D,L,C,m_glob1,m_glob2,m_glob3,m_glob4,s_glob1,s_glob2,s_glob3,s_glob4,disp,orthog)
    thetas = [sim_param(L,L,0.0,m2,m3,1.0,s2,s3,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def sim_param_latmix_3(K,D,L,m2=0.0,m3=0.0,s2=1.0,s3=1.0,m_glob1=0.0,m_glob2=0.0,s_glob1=1.0,s_glob2=1.0,la=1.0,p=0.1,disp=False,orthog=True):
    eta = sim_param_LRPSI(D,L,m_glob1,m_glob2,s_glob1,s_glob2,la,p,disp,orthog)
    thetas = [sim_param(L,L,0.0,m2,m3,1.0,s2,s3,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def sim_omega(N,K,N_min=2):
    
    if N < K*N_min:
        K = int(N/N_min)
    
    nb_random = N - K*N_min
    random_part = rd.choice(K,nb_random)
    sorted_omega = np.concatenate([np.concatenate([k*np.ones(N_min) for k in range(K)]),random_part])
    omega = rd.permutation(sorted_omega)
    return omega.astype(int)

def sim_data_obsmix_1(thetas,N,N_min=0,disp=False):
    
    K = len(thetas)
    D,L = np.shape(thetas[0][0])
    
    omega = sim_omega(N,K,int(max(L+1,N_min)))
    K = int(max(omega)) + 1
    
    Z = rd.normal(0,1,(N,L))
    Y = np.array([thetas[omega[n]][0]@Z[n] + thetas[omega[n]][1] + rd.normal(0,thetas[omega[n]][2],D) for n in range(N)])
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\omega = $', omega)
    return Z, omega, Y

def sim_data_obsmix_2(thetas,N,N_min=0,Sigma_X=None,disp=False):
    
    K = len(thetas)
    
    D1,L = np.shape(thetas[0][0])
    D2,C = np.shape(thetas[0][1])
    
    omega = sim_omega(N,K,int(max(L+1,C+1,N_min)))
    K = int(max(omega)) + 1
    
    if D1 != D2 :
        print('Erreur de dimension sur W et V')
    else :
        D = D1
    
    Z = rd.normal(0,1,(N,L))
    if type(Sigma_X) == type(None) :
        X = rd.normal(0,1,(N,C))
    else :
        X = rd.multivariate_normal(np.zeros(C),Sigma_X,N)
    
    Y = np.array([thetas[omega[n]][0]@Z[n] + thetas[omega[n]][1]@X[n] + thetas[omega[n]][2] + rd.normal(0,thetas[omega[n]][3],D) for n in range(N)])
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
        
    return Z, omega, Y, X

def sim_data_obsmix_3(thetas,N,N_min=0,disp=False):
    
    K = len(thetas)
    
    D,L = np.shape(thetas[0][0])
    D1,D2 = np.shape(thetas[0][1])
    if D1 != D2 or D1 != D :
        print('Erreur de dimension sur W et Lambda')
    
    omega = sim_omega(N,K,int(max(L+1,N_min)))
    K = int(max(omega)) + 1
    
    Z = rd.normal(0,1,(N,L))
    X = np.array([rd.multivariate_normal(np.zeros(D),nla.inv(thetas[omega[n]][1])) for n in range(N)])
    Y = np.array([thetas[omega[n]][0]@Z[n] + X[n] + rd.normal(0,thetas[omega[n]][2],D) for n in range(N)])
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
        
    return Z, omega, Y, X

def sim_data_latmix_1(eta,thetas,N,N_min=2,disp=False):
    
    W, mu, sigma2 = eta
    D,L = np.shape(W)
    K = len(thetas)
    
    omega = sim_omega(N,K,N_min)
    K = int(max(omega)) + 1
    
    Z_orig = rd.normal(0,1,(N,L))
    Z = np.zeros((N,L))
    for n in range(N):
        k = omega[n]
        mu_k,sigma2_k = thetas[k]
        Z[n] = np.sqrt(sigma2_k)*Z_orig[n] + mu_k
        
    noise = rd.normal(0,sigma2,(N,D))
    Y = Z@np.transpose(W) + mu + noise
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\omega = $', omega)
    
    return Z, omega, Y

def sim_data_latmix_2(eta,thetas,N,N_min=2,Sigma_X=None,disp=False):
    
    W, V, mu, sigma2 = eta
    D,L = np.shape(W)
    D2,C = np.shape(V)
    K = len(thetas)
    
    omega = sim_omega(N,K,N_min)
    K = int(max(omega)) + 1
    
    Z_orig = rd.normal(0,1,(N,L))
    Z = np.zeros((N,L))
    for n in range(N):
        k = omega[n]
        mu_k,sigma2_k = thetas[k]
        Z[n] = np.sqrt(sigma2_k)*Z_orig[n] + mu_k
        
    if type(Sigma_X) == type(None) :
        X = rd.normal(0,1,(N,C))
    else :
        X = rd.multivariate_normal(np.zeros(C),Sigma_X,N)
    noise = rd.normal(0,sigma2,(N,D))
    
    Y = Z@np.transpose(W) + X@np.transpose(V) + mu + noise
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
    
    return Z, omega, Y, X

def sim_data_latmix_3(eta,thetas,N,N_min=2,Sigma_X=None,disp=False):
    
    W, Lambda, mu, sigma2 = eta
    D,L = np.shape(W)
    D1,D2 = np.shape(Lambda)
    K = len(thetas)
    
    omega = sim_omega(N,K,N_min)
    K = int(max(omega)) + 1
    
    Z_orig = rd.normal(0,1,(N,L))
    Z = np.zeros((N,L))
    for n in range(N):
        k = omega[n]
        mu_k,sigma2_k = thetas[k]
        Z[n] = np.sqrt(sigma2_k)*Z_orig[n] + mu_k
        
    X = np.array([rd.multivariate_normal(np.zeros(D),nla.inv(Lambda))])
    noise = rd.normal(0,sigma2,(N,D))
    
    Y = Z@np.transpose(W) + X + mu + noise
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
    
    return Z, omega, Y, X

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions d'estimation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def L_opt(S,beta=0.5,L_min=1,detail=False):
    
    D1,D2 = np.shape(S)
    if D1 != D2:
        print('Erreur de dimensions sur S')
    else :
        D = D1
        SpS,P = nla.eig(S)
        
        ordre = np.sort(SpS)
        if L_min <= 1:
            diff_ordre = ordre[1:] - ordre[:-1]
            L_star = D - int(np.argmax(diff_ordre)) - 1
        else :
            diff_ordre = ordre[1:1-L_min] - ordre[:-L_min]
            L_star = D - int(np.argmax(diff_ordre)) - 1
        
        total_span = ordre[-1]-ordre[0]
        taken_span = ordre[D-L_star]-ordre[D-L_star-1]
        
        coeff_incert = np.log(total_span/taken_span)/np.log(D)
        
        if detail :
            plt.figure()
            plt.step(np.arange(D),ordre)
            plt.plot([D-L_star-1,D-L_star-1],[ordre[D-L_star-1],ordre[D-L_star]])
            plt.plot([D-L_star-1,D-L_star-1],[ordre[D-L_star-1],0],'--',label='$D-L_{star}$')
            plt.legend()
            plt.title('Spectre ordonné de la matrice de covariance')
            plt.show()
            
            print("Le coefficient d'incertitude est de",coeff_incert)
        
        return L_star

def L_opt_2(S,beta=0.5,L_min=1,detail=False):
    
    D1,D2 = np.shape(S)
    if D1 != D2:
        print('Erreur de dimensions sur S')
    else :
        D = D1
        SpS,P = nla.eig(S)
        
        ordre = np.sort(SpS)
        if L_min <= 1:
            L_star = D - R_elbow(ordre,beta) - 1
        else :
            L_star = D - R_elbow(ordre[:1-L_min],beta) - 1
        
        total_span = ordre[-1]-ordre[0]
        taken_span = ordre[D-L_star]-ordre[D-L_star-1]
        
        coeff_incert = np.log(total_span/taken_span)/np.log(D)
        
        if detail :
            plt.figure()
            plt.step(np.arange(D),ordre)
            plt.plot([D-L_star-1,D-L_star-1],[ordre[D-L_star-1],ordre[D-L_star]],label='$D-L_{star}$')
            plt.plot([D-L_star-1,D-L_star-1],[ordre[D-L_star-1],0],'--',label='$D-L_{star}$')
            plt.legend()
            plt.title('Spectre ordonné de la matrice de covariance')
            plt.show()
            
            print("Le coefficient d'incertitude est de",coeff_incert)
        
        return L_star

def L_opt_3(S,beta=0.5,L_min=1,detail=False):
    
    D1,D2 = np.shape(S)
    if D1 != D2:
        print('Erreur de dimensions sur S')
    else :
        D = D1
        SpS,P = nla.eig(S)
        
        ordre = np.sort(SpS)
        inertie = np.cumsum(ordre)
        
        if L_min <= 1:
            L_star = D - R_elbow(inertie,beta) - 1
        else :
            L_star = D - R_elbow(inertie[:1-L_min],beta) - 1
        
        total_span = ordre[-1]-ordre[0]
        taken_span = ordre[D-L_star]-ordre[D-L_star-1]
        
        coeff_incert = np.log(total_span/taken_span)/np.log(D)
        
        if detail :
            plt.figure()
            plt.step(np.arange(D),ordre)
            plt.plot([D-L_star-1,D-L_star-1],[ordre[D-L_star-1],ordre[D-L_star]],label='$D-L_{star}$')
            plt.plot([D-L_star-1,D-L_star-1],[ordre[D-L_star-1],0],'--',label='$D-L_{star}$')
            plt.legend()
            plt.title('Spectre ordonné de la matrice de covariance')
            plt.show()
            
            print("Le coefficient d'incertitude est de",coeff_incert)
        
        return L_star

def PCA(Y,L=None,orthog=True):
    
    N,D = np.shape(Y)
    
    #Centrage de Y
    mu_ML = np.mean(Y,axis=0)
    Yc = np.array([y-mu_ML for y in Y])
    
    #Diagonalisation de S, choix des axes principaux
    S = 1/N * np.transpose(Yc)@Yc
    
    if type(L) == type(None):
        L = L_opt(S)
    
    SpS, P = nla.eig(S)
    ordre = np.sort(SpS)
    inlist = [k for k in range(D) if SpS[k] in ordre[D-L:]]
    
    #Estimation de W
    tW = np.array([P[:,k] for k in inlist])
    W = np.transpose(tW)
    if orthog :
        W = orthogonalize(W)
        tW = np.transpose(W)
    
    #Estimation de Z
    Z = Yc @ W @ nla.inv(tW@W)
    
    return S, W, Z

def bruit(S,L):
    D1,D2 = np.shape(S)
    if D1 != D2 or D1 <= L or D2 <= L :
        print('Erreur de dimension sur S')
    else :
        D = D1
        SpS, P = nla.eig(S)
        sigma2 = np.mean(np.sort(SpS)[D-L:])
        return sigma2

def PPCA_EM(Y,L=None,nb_steps=1000,err=0.0,orthog=True,tempo=True):
    
    N,D = np.shape(Y)
    
    #Centrage de Y
    mu_Y = np.mean(Y,axis=0)
    Yc = np.array([y-mu_Y for y in Y])
    
    #Initialisation
    S, W, Z = PCA(Y,L,orthog)
    
    if type(L) == type(None):
        L = L_opt(S)
    
    #Estimation de sigma²
    sigma2 = bruit(S,L)
    
    #Algorithme E-M
    dist = err+1
    t = 0
    while dist > err and t < nb_steps :
        new_Z = Yc @ W @ nla.inv(np.transpose(W)@W)
        new_W = np.transpose(Yc) @ new_Z @ nla.inv(np.transpose(new_Z) @ new_Z)
        dist = np.sum((Z-new_Z)**2) + np.sum((W-new_W)**2)
        W = new_W
        Z = new_Z
        t += 1
    if tempo :
        print('t = ',t)
    
    return W, Z, sigma2

def PPCA(Y,L=None,orthog=True):
    
    N,D = np.shape(Y)
    
    #Centrage de Y
    mu_ML = np.mean(Y,axis=0)
    Yc = np.array([y-mu_ML for y in Y])
    
    #Estimation de sigma²
    S = 1/N * np.transpose(Yc)@Yc
    
    if type(L) == type(None):
        L = L_opt(S)
    
    SpS, P = nla.eig(S)
    ordre = np.sort(SpS)
    sigma2 = np.mean(ordre[:D-L])
    inlist = [k for k in range(D) if SpS[k] in ordre[D-L:]]    
    
    #Estimation de W et Z
    tU_L = np.array([P[:,k] for k in inlist])
    L_L = np.diag(ordre[D-L:])-sigma2*np.eye(L)
    tW = L_L @ tU_L
    W = np.transpose(tW)
    Z = Yc @ W @ nla.inv(tW@W)
    
    return W, Z, sigma2

def V_estimator(Y,X,Sigma):
    V = np.transpose(nla.inv(np.transpose(X)@nla.inv(Sigma)@X) @ (np.transpose(X)@nla.inv(Sigma)@Y))
    return V

def Sigma_estimator(Y,X,V,L):
    S, W, Z = PCA(Y - X@np.transpose(V), L)
    N,C = np.shape(X)
    sigma2 = bruit(S,L)
    Sigma = X @ np.transpose(X) + sigma2 * np.eye(N)
    return Sigma
    
def ML_RCA(Y,X,L,V=None,Sigma=None,nb_steps=1000,err=0.0,tempo=True):
    
    N1,D = np.shape(Y)
    N2,C = np.shape(X)
    
    if N1 != N2 :
        print('Erreur de dimensions sur Y et X')
        S, W, Z = PCA(Y,L)
        sigma2 = bruit(S,L)
        return W, Z, sigma2
    else :
        N = N1
    
    #Centrage de Y et X
    mY = np.mean(Y,axis=0)
    mX = np.mean(X,axis=0)
    Yc = np.array([y-mY for y in Y])
    Xc = np.array([x-mX for x in X])
    
    #Initialisation
    
    #Si V est donné
    if type(V) != type(None) :
        
        #Copie de V s'il est donné
        V_hat = V.copy()
        
        if type(Sigma) != type(None) :
            
            #Copie de Sigma s'il est donné
            Sigma_hat = Sigma.copy()
        else :
            #Estimation de Sigma à partir de V si V est donné mais pas Sigma
            Sigma_hat = Sigma_estimator(Yc,Xc,V,L)
    
    #Si V n'est pas donné
    if type(V) == type(None) :
        
        if type(Sigma) != type(None) :
            
            #Copie de Sigma s'il est donné
            Sigma_hat = Sigma.copy()
            #Estimation de V à partir de Sigma si Sigma est donné mais pas V
            V_hat = V_estimator(Yc,Xc,Sigma)
        
        #Estimation de V et Sigma s'ils ne sont pas donnés
        else :
            
            #Initialisation
            S, W, Z = PCA(Yc,L)
            sigma2 = bruit(S,L)
            Sigma_hat = Xc@np.transpose(Xc) + sigma2 * np.eye(N)
            V_hat = V_estimator(Yc,Xc,Sigma_hat)
            
            dist = err+1
            t = 0
            
            #Boucle
            while dist > err and t < nb_steps :
                new_Sigma = Sigma_estimator(Yc,Xc,V_hat,L)
                new_V = V_estimator(Yc,Xc,new_Sigma)
                dist = np.sum((V_hat-new_V)**2) + np.sum((Sigma_hat-new_Sigma)**2)
                V_hat = new_V
                Sigma_hat = new_Sigma
                t += 1
            if tempo :
                print('t =',t)
    
    D1,C1 = np.shape(V_hat)
    if D != D1 or C != C1 :
        print('Erreur de dimensions sur V')
        S, W, Z = PCA(Yc,L)
        sigma2 = bruit(S,L)
        return W, Z, V_hat, Sigma_hat, sigma2
    
    N1,N2 = np.shape(Sigma_hat)
    if N != N1 or N1 != N2 :
        print('Erreur de dimensions sur Sigma')
        S, W, Z = PCA(Yc,L)
        sigma2 = bruit(S,L)
        return W, Z, V_hat, Sigma_hat, sigma2
    
    #Estimation de Z
    S_hac = 1/D * Yc @ np.transpose(Yc)
    A = nla.inv(Sigma_hat) @ S_hac
    SpA, P1 = nla.eig(A)
    ord_A = np.sort(SpA)
    inlist_A = [k for k in range(N) if SpA[k] in  ord_A[N-L:]]
    P_Z = np.transpose(np.array([P1[:,k] for k in inlist_A]))
    D_Z = np.diag(np.array([SpA[k] for k in inlist_A]))
    Z = Sigma_hat @ P_Z
    
    #Estimation de W
    W = (np.transpose(Yc)-V_hat@np.transpose(Xc)) @ Z @ nla.inv(np.transpose(Z)@Z)
    
    return W, Z, V_hat, Sigma_hat, sigma2

def ML_RCA_2(Y,X,L,V=None,Sigma=None,nb_steps=100,err=0.0,tempo=True):
    
    N1,D = np.shape(Y)
    N2,C = np.shape(X)
    
    if N1 != N2 :
        print('Erreur de dimensions sur Y et X')
        S, W, Z = PCA(Y,L)
        sigma2 = bruit(S,L)
        return W, Z, sigma2
    else :
        N = N1
    
    #Centrage de Y et X
    mY = np.mean(Y,axis=0)
    mX = np.mean(X,axis=0)
    Yc = np.array([y-mY for y in Y])
    Xc = np.array([x-mX for x in X])
    
    #Initialisation
    
    #Si V est donné
    if type(V) != type(None) :
        
        #Copie de V s'il est donné
        V_hat = V.copy()
        
        if type(Sigma) != type(None) :
            
            #Copie de Sigma s'il est donné
            Sigma_hat = Sigma.copy()
        else :
            #Estimation de Sigma à partir de V si V est donné mais pas Sigma
            Sigma_hat = Sigma_estimator(Yc,Xc,V,L)
            
        D1,C1 = np.shape(V_hat)
        if D != D1 or C != C1 :
            print('Erreur de dimensions sur V')
            S, W, Z = PCA(Yc,L)
            sigma2 = bruit(S,L)
            return W, Z, V_hat, Sigma_hat, sigma2
            
        N1,N2 = np.shape(Sigma_hat)
        if N != N1 or N1 != N2 :
            print('Erreur de dimensions sur Sigma')
            S, W, Z = PCA(Yc,L)
            sigma2 = bruit(S,L)
            return W, Z, V_hat, Sigma_hat, sigma2
    
        #Estimation de Z
        S_hac = 1/D * Yc @ np.transpose(Yc)
        A = nla.inv(Sigma_hat) @ S_hac
        SpA, P1 = nla.eig(A)
        ord_A = np.sort(SpA)
        inlist_A = [k for k in range(N) if SpA[k] in  ord_A[N-L:]]
        P_Z = np.transpose(np.array([P1[:,k] for k in inlist_A]))
        D_Z = np.diag(np.array([SpA[k] for k in inlist_A]))
        Z = Sigma_hat @ P_Z
    
        #Estimation de W
        W = (np.transpose(Yc)-V_hat@np.transpose(Xc)) @ Z @ nla.inv(np.transpose(Z)@Z)
    
    #Si V n'est pas donné
    if type(V) == type(None) :
        
        if type(Sigma) != type(None) :
            
            #Copie de Sigma s'il est donné
            Sigma_hat = Sigma.copy()
            #Estimation de V à partir de Sigma si Sigma est donné mais pas V
            V_hat = V_estimator(Yc,Xc,Sigma)
            
            D1,C1 = np.shape(V_hat)
            if D != D1 or C != C1 :
                print('Erreur de dimensions sur V')
                S, W, Z = PCA(Yc,L)
                sigma2 = bruit(S,L)
                return W, Z, V_hat, Sigma_hat, sigma2
                
            N1,N2 = np.shape(Sigma_hat)
            if N != N1 or N1 != N2 :
                print('Erreur de dimensions sur Sigma')
                S, W, Z = PCA(Yc,L)
                sigma2 = bruit(S,L)
                return W, Z, V_hat, Sigma_hat, sigma2
            
            #Estimation de Z
            S_hac = 1/D * Yc @ np.transpose(Yc)
            A = nla.inv(Sigma_hat) @ S_hac
            SpA, P1 = nla.eig(A)
            ord_A = np.sort(SpA)
            inlist_A = [k for k in range(N) if SpA[k] in  ord_A[N-L:]]
            P_Z = np.transpose(np.array([P1[:,k] for k in inlist_A]))
            D_Z = np.diag(np.array([SpA[k] for k in inlist_A]))
            Z = Sigma_hat @ P_Z
            
            #Estimation de W
            W = (np.transpose(Yc)-V_hat@np.transpose(Xc)) @ Z @ nla.inv(np.transpose(Z)@Z)
        
        #Estimation de V et Sigma s'ils ne sont pas donnés
        else :
            
            #Initialisation
            S, W, Z = PCA(Yc,L)
            sigma2 = bruit(S,L)
            Sigma_hat = Xc@np.transpose(Xc) + sigma2 * np.eye(N)
            V_hat = V_estimator(Yc,Xc,Sigma_hat)
            
            #Estimation de Z
            S_hac = 1/D * Yc @ np.transpose(Yc)
            A = nla.inv(Sigma_hat) @ S_hac
            SpA, P1 = nla.eig(A)
            ord_A = np.sort(SpA)
            inlist_A = [k for k in range(N) if SpA[k] in  ord_A[N-L:]]
            P_Z = np.transpose(np.array([P1[:,k] for k in inlist_A]))
            D_Z = np.diag(np.array([SpA[k] for k in inlist_A]))
            Z = Sigma_hat @ P_Z
            
            #Estimation de W
            W = (np.transpose(Yc)-V_hat@np.transpose(Xc)) @ Z @ nla.inv(np.transpose(Z)@Z)
            
            dist = err+1
            t = 0
            
            #Boucle
            while dist > err and t < nb_steps :
                
                #Estimation de Z
                S_hac = 1/D * Yc @ np.transpose(Yc)
                A = nla.inv(Sigma_hat) @ S_hac
                SpA, P1 = nla.eig(A)
                ord_A = np.sort(SpA)
                inlist_A = [k for k in range(N) if SpA[k] in  ord_A[N-L:]]
                P_Z = np.transpose(np.array([P1[:,k] for k in inlist_A]))
                D_Z = np.diag(np.array([SpA[k] for k in inlist_A]))
                new_Z = Sigma_hat @ P_Z
            
                #Estimation de W
                new_W = (np.transpose(Yc)-V_hat@np.transpose(Xc)) @ new_Z @ nla.inv(np.transpose(new_Z)@new_Z)
                
                #Estimation de V
                new_V = np.transpose(Yc-new_Z@np.transpose(new_W)) @ Xc @ nla.inv(np.transpose(Xc)@Xc)
                
                #Estimation de Sigma
                sigma_2_hat = np.mean((Yc-new_Z@np.transpose(new_W)-Xc@np.transpose(new_V))**2)
                new_Sigma = Xc @ np.transpose(Xc) + sigma_2_hat * np.eye(N)
                
                dist = np.sum((Z@np.transpose(W)-new_Z@np.transpose(new_W))**2) + np.sum((V_hat-new_V)**2) + np.sum((Sigma_hat-new_Sigma)**2)
                W = new_W
                Z = new_Z
                V_hat = new_V
                Sigma_hat = new_Sigma
                sigma2 = sigma_2_hat
                t += 1
                
            if tempo :
                print('t =',t)
    
    return W, Z, V_hat, Sigma_hat, sigma2

def tXX_estimator(Y,W,sigma2,Lambda):
    
    D,L = np.shape(W)
    D1,D2 = np.shape(Lambda)
    N,D3 = np.shape(Y)
    
    if D != D1 or D != D2 or D != D3 :
        print('Erreur de dimensions sur Lambda, Y ou W')
    else :
        qty_1 = nla.inv(nla.inv(W@np.transpose(W) + sigma2*np.eye(D)) + Lambda)
        qty_2 = np.array([qty_1 @ nla.inv(W@np.transpose(W) + sigma2*np.eye(D)) @ y for y in Y])
        tXX = np.array([qty_1 + np.transpose(np.array([x]))@np.array([x]) for x in qty_2])
        return tXX

def Lambda_estimator(tXX,la):
    
    N,D1,D2 = np.shape(tXX)
    
    if D1 != D2 :
        print('Erreur de dimensions sur tXX')
    else :
        D = D1
        S = np.mean(tXX,axis=0)
        Lambda, st_else = sklcov.graphical_lasso(S,la,max_iter=50)
        return Lambda
        
def W_estimator(Y,Lambda,sigma2,L):
    
    N,D = np.shape(Y)
    D1,D2 = np.shape(Lambda)
    
    if D != D1 or D != D2 :
        print('Erreur de dimensions sur Y ou Lambda')
    else :
        Sigma = nla.inv(Lambda) + sigma2*np.eye(D)
        S_hat = 1/N * np.transpose(Y)@Y
        A = nla.inv(Sigma) @ S_hat
        
        #GEP
        SpA, P = nla.eig(A)
        ord_A = np.sort(SpA)
        inlist_A = [k for k in range(D) if SpA[k] in ord_A[D-L:]]
        P_W = normalize(np.transpose(np.array([P[:,k] for k in inlist_A])))
        D_W = np.diag(np.array([SpA[k] for k in inlist_A]))
        W = Sigma @ P_W
        
        return W

def log_p_LRPSI(Y,W,Lambda,sigma2,la):
    
    N,D = np.shape(Y)
    D1,L = np.shape(W)
    D2,D3 = np.shape(Lambda)
    
    if D != D1 or D!= D2 or D != D3 :
        print('Erreur de dimensions sur Y, W ou Lambda')
    else :
        Cov = W @ np.transpose(W) + nla.inv(Lambda) + sigma2*np.eye(D)
        inv_Cov = nla.inv(Cov)
        qty = np.sum(np.array([-np.log(np.abs(nla.det(Cov))) - 1/2 * np.array([y])@inv_Cov@y - la*np.sum(np.abs(Lambda)) for y in Y]))
        return qty

def EM_RCA_LRPSI(Y,X,L,Lambda=None,la=None,sigma2=None,nb_steps=1000,err=0.0,tempo=True):
    
    N1,D1 = np.shape(Y)
    N2,D2 = np.shape(X)
    
    if N1 != N2 or D1 != D2 :
        print('Erreur de dimensions sur Y et X')
        S, W, Z = PCA(Y,L)
        sigma2 = bruit(S,L)
        return W, Z, sigma2
    else :
        N = N1
        D = D1
    
    #Centrage de Y et X
    mY = np.mean(Y,axis=0)
    mX = np.mean(X,axis=0)
    Yc = np.array([y-mY for y in Y])
    Xc = np.array([x-mX for x in X])
    
    #Initialisation
    
    #Estimation de Lambda s'il n'est pas donné
    if type(Lambda) == type(None):
        covX = 1/N * np.transpose(Xc)@Xc
        Lambda_hat = nla.inv(covX)
    else :
        Lambda_hat = Lambda.copy()
    
    D1,D2 = np.shape(Lambda_hat)
    if D != D1 or D != D2 :
        print('Erreur de dimensions sur Lambda')
        S, W, Z = PCA(Yc,L)
        sigma2 = bruit(S,L)
        return W, Z, Lambda_hat, sigma2
    
    #Estimation de lambda s'il n'est pas donné
    if type(la) == type(None):
        la_hat = D**2 * 2/trace(Lambda_hat)
        calc_la = True
    else :
        la_hat = la.copy()
        calc_la = False
     
    #Estimation de sigma2 s'il n'est pas donné
    if type(sigma2) == type(None):
        S,W_hat,Z = PCA(Yc-Xc,L)
        sigma2 = bruit(S,L)
    
    #Boucle EM/RCA pour estimer Z
    gain = err + 1
    t = 0
    W_hat = W_estimator(Yc,Lambda_hat,sigma2,L)
    log_p = log_p_LRPSI(Yc,W_hat,Lambda_hat,sigma2,la_hat)
    
    while gain > err and t < nb_steps :
        
        #E-step :
        new_tXX = tXX_estimator(Yc,W_hat,sigma2,Lambda_hat)
        
        #M-step :
        new_Lambda = Lambda_estimator(new_tXX,la_hat)
        
        #RCA-step :
        new_W = W_estimator(Yc,new_Lambda,sigma2,L)
        
        #Vérification :
        new_log_p = log_p_LRPSI(Yc,new_W,new_Lambda,sigma2,la_hat)
        gain = new_log_p - log_p
        
        if gain >= 0 :
            Lambda_hat = new_Lambda
            W_hat = new_W
            if calc_la :
                la_hat = D**2 * 2/trace(Lambda_hat)
        
        t += 1
        log_p = new_log_p
        
    if tempo :
        print('t =',t)
    
    #Estimation de Z
    Z = (Yc - Xc) @ W_hat @ nla.inv(np.transpose(W_hat)@W_hat)
    
    return W_hat,Z,Lambda_hat,sigma2

def MLE_Gauss(Y):
    N,D = np.shape(Y)
    mu_ML = np.mean(Y,axis=0)
    sigma2_ML = np.var(Y)*(D/(D-1))
    return mu_ML,sigma2_ML

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions utiles pour le clustering
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Single-linkage
def d_SL(Pdm,G,H):
    if np.any(Pdm-np.transpose(Pdm)) or np.any(Pdm<0):
        print("Pdm n'est pas une matrice de dissimilarité")
    else :
        N,N1 = np.shape(Pdm)
        dist_tab = np.array([[Pdm[i][j] for j in H] for i in G])
        return np.min(dist_tab)

#Complete-linkage
def d_CL(Pdm,G,H):
    if np.any(Pdm-np.transpose(Pdm)) or np.any(Pdm<0):
        print("Pdm n'est pas une matrice de dissimilarité")
    else :
        N,N1 = np.shape(Pdm)
        dist_tab = np.array([[Pdm[i][j] for j in H] for i in G])
        return np.max(dist_tab)

#Average-linkage
def d_AL(Pdm,G,H):
    if np.any(Pdm-np.transpose(Pdm)) or np.any(Pdm<0):
        print("Pdm n'est pas une matrice de dissimilarité")
    else :
        N,N1 = np.shape(Pdm)
        dist_tab = np.array([[Pdm[i][j] for j in H] for i in G])
        return np.mean(dist_tab)

#Distance Ward
def d_L2_Ward(X,G,H):
    
    N,D = np.shape(X)
    
    if max(G) >= N or max(H) >= N :
        print("Pas assez d'individus")
    else :
        mu_G = np.mean(np.array([X[n] for n in G]),axis=0)
        mu_H= np.mean(np.array([X[n] for n in H]),axis=0)
        return np.sum((mu_G - mu_H)**2)

#Distance L2
def dissim_L2(X):
    N,D = np.shape(X)
    return np.array([[np.sqrt(np.sum((X[i]-X[j])**2)) for i in range(N)] for j in range(N)])

#Condensation d'une matrice de dissimilarité
def condense(PdM):
    N,N1 = np.shape(PdM)
    return np.concatenate([PdM[n][n+1:] for n in range(N-1)])

#Triage des vecteurs
def tri(X,omega,K=None):
    
    N = len(omega)
    if type(K) == type(None):
        K = int(max(omega) + 1)
        
    for k in range(K):
        if k not in omega :
            print(k)
            print("tri : Clusters vides")
    
    tri_X = [np.array([]) for k in range(K)]
    
    for k in range(K):
        tri_X[k] = np.array([X[n] for n in range(N) if omega[n]==k])
    
    return tri_X

def omegate(clusters):
    
    K = len(clusters)
    N = max([max(clus) for clus in clusters]) + 1
    omega = np.zeros(N)
    
    for k in range(K) :
        for n in clusters[k] :
            omega[n] = k
    
    return omega.astype(int)

def matrixage(omega):
    
    N = len(omega)
    K = int(np.max(omega) + 1)
    O = np.array([[int(omega[n] == k) for k in range(K)] for n in range(N)])
    
    return O

#Occurences
def occurences(omega):
    
    N = len(omega)
    K = int(np.max(omega)) + 1
    
    occur = np.zeros(K)
    for n in range(N):
        occur[omega[n]] += 1
    
    return occur.astype(int)

def perm_opt(omega1,omega2):
    
    N1 = len(omega1)
    N2 = len(omega2)
    occ1 = occurences(omega1)
    occ2 = occurences(omega2)
    K1 = len(occ1)
    K2 = len(occ2)
    
    if N1 != N2 or K1 != K2 or np.any(occ1==0) or np.any(occ2==0):
        print("Les omegas ne correspondent pas")
    else:
        N = N1
        K = K1
        
        O1 = matrixage(omega1)
        O2 = matrixage(omega2)
        R_occ = np.transpose(O1)@O2
        R = np.diag(1/occ1) @ R_occ
        
        sure=[]
        unsure=list(np.arange(K))
        taken=[]
        untaken=list(np.arange(K))
        
        s_sure = -np.ones(K)
        
        for k in range(K):
            s_k = int(np.argmax(R[k]))
            if k == int(np.argmax(R[:,s_k])):
                sure.append(k)
                unsure.remove(k)
                taken.append(s_k)
                untaken.remove(s_k)
                s_sure[k] = s_k
        
        nb_unsure = len(unsure)
        
        errs = []
        perms = list(itt.permutations(range(nb_unsure)))
        for perm in perms :
            s_test = s_sure
            for i in range(nb_unsure):
                j = perm[i]
                s_test[unsure[i]] = untaken[j]
            s_omega1 = np.array([s_test[o] for o in omega1]).astype(int)
            err = np.sum(((s_omega1 - omega2).astype(bool)).astype(float))
            errs.append(err)
        
        ind_opt = int(np.argmin(np.array(errs)))
        us_opt = perms[ind_opt]
        s_opt = s_sure
        for i in range(nb_unsure):
            j = perm[i]
            s_opt[unsure[i]] = untaken[j]
        
        return s_opt.astype(int)

#Distance to closest point
def Dist_CP(x,M):
    t,D = np.shape(M)
    D1 = len(x)
    if D != D1:
        print("x et les moyennes n'ont pas même dimensions")
    else :
        return min([np.sum((x-M[k])**2) for k in range(t)])

#Silhouette coefficient
def sil_coeff(n,X,omega):
    
    x = X[n]
    N = len(X)
    K = int(max(omega)+1)
    
    M = np.array([np.mean(np.array([X[j] for j in range(n) if omega[j]==k])) for k in range(K)])
    
    k_star = omega[n]
    dists = np.array([np.sum((x-M[k])**2) for k in range(K)])
    dists[k_star] += np.max(dists) + 1
    k_prime = np.argmin(dists)
    
    a = np.mean(np.array([np.sum((x-X[j])**2) for j in range(N) if omega[j] == k_star]))
    b = np.mean(np.array([np.sum((x-X[j])**2) for j in range(N) if omega[j] == k_prime]))
    
    return (b-a)/(max(a,b))

#Silhouette score
def sil_score(X,omega):
    N = len(X)
    return np.mean([sil_coeff(n,X,omega) for n in range(N)])

#Distorsion
def distorsion(X,omega):
    tri_X = tri(X,omega)
    return np.sum(np.array([np.sum(np.var(x,axis=0)) for x in tri_X]))

def Lap(Psm):
    
    N,N1 = np.shape(Psm)
    
    if np.any(Psm-np.transpose(Psm)) or np.any(Psm<0):
        print("Psm n'est pas une matrice de poids symétrique")
    else :
        vec_D = np.sum(Psm, axis=0)
        rinv_D = np.diag(1/np.sqrt(vec_D))
        
        L = np.eye(N) - rinv_D @ Psm @ rinv_D
        
        return L

def ARS(omega1,omega2):
    
    return sklmc.adjusted_rand_score(omega1,omega2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions pour reconstruire ou représenter les données
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def PCA_rec(W,Z,mu):
    Y = Z @ np.transpose(W) + mu
    return Y

def RCA_rec(W,Z,V,X,mu):
    Xc = X - np.mean(X,axis=0)
    Y = Z @ np.transpose(W) + Xc @ np.transpose(V) + mu
    return Y

def MFA_rec_1(thetas,Z,omega):
    
    D,L = np.shape(thetas[0][0])
    N = len(omega)
    Y = np.zeros((N,D))
    
    for n in range(N):
        W,mu,sigma2 = thetas[omega[n]]
        z = Z[n]
        Y[n] = W@z + mu
    
    return Y

def MFA_rec_2(thetas,Z,X,omega):
    
    D,L = np.shape(thetas[0][0])
    N = len(omega)
    Y = np.zeros((N,D))
    tri_X = tri(X,omega)
    
    for n in range(N):
        W,V,mu,sigma2 = thetas[omega[n]]
        z = Z[n]
        x = X[n] - np.mean(tri_X[omega[n]],axis=0)
        
        Y[n] = W@z + V@x + mu
    
    return Y

def CA_graph(Y,Y_hat):
    
    N,D = np.shape(Y)
    
    for j in range(int(D/2)):
        
        plt.figure()
        
        plt.scatter(Y[:,2*j],Y[:,2*j+1],label='$Y$')
        plt.scatter(Y_hat[:,2*j],Y_hat[:,2*j+1],label='$\hat{Y}$')
        
        for n in range(N):
            plt.plot([Y[n][2*j],Y_hat[n][2*j]],[Y[n][2*j+1],Y_hat[n][2*j+1]],color='black')
            
        plt.legend()
        plt.show()

def MFA_graph(Y,Y_hat,omega_hat,omega=None,labels=None):
    
    N,D = np.shape(Y)
    K = int(max(omega_hat) + 1)
    
    if type(omega) == type(None):
        s_omega = omega_hat
        K = int(max(omega_hat) + 1)
    else:
        s_opt = perm_opt(omega,omega_hat)
        s_omega = np.array([s_opt[k] for k in omega]).astype(int)
        K = int(max(omega_hat) + 1)
        
    tri_Y = tri(Y,s_omega,K)
    tri_Y_hat = tri(Y_hat,omega_hat,K)
    colors_orig = ['#802020','#E08080','#208020','#80E080','#202080','#8080E0','#808020','#E0E080','#802080','#E080E0','#208080','#80E0E0','#805020','#E0B080','#508020','#B0E080','#208050','#80E0B0','#205080','#80B0E0','#502080','#B080E0','#802050','#E080B0','#202020','#E0E0E0']
    nb_cyc = int(np.ceil(K/len(colors_orig)))
    colors = colors_orig*nb_cyc
    
    if type(labels) == type(None):
        for j in range(int(D/2)):
            
            plt.figure()
            
            for k in range(K):
                if len(tri_Y[k]) > 0:
                    plt.scatter(tri_Y[k][:,2*j],tri_Y[k][:,2*j+1],label='$Y$',color=colors[2*k])
                if len(tri_Y_hat[k]) > 0:
                    plt.scatter(tri_Y_hat[k][:,2*j],tri_Y_hat[k][:,2*j+1],label='$\hat{Y}$',color=colors[2*k+1])
            
            for n in range(N):
                plt.plot([Y[n][2*j],Y_hat[n][2*j]],[Y[n][2*j+1],Y_hat[n][2*j+1]],color='black')
            
            plt.legend()
            plt.show()
    else :
        for j in range(int(D/2)):
            
            plt.figure()
            
            for k in range(K):
                if len(tri_Y[k]) > 0:
                    plt.scatter(tri_Y[k][:,2*j],tri_Y[k][:,2*j+1],label='$Y$ - '+labels[k],color=colors[2*k])
                if len(tri_Y_hat[k]) > 0:
                    plt.scatter(tri_Y_hat[k][:,2*j],tri_Y_hat[k][:,2*j+1],label='$\hat{Y} - $'+labels[k],color=colors[2*k+1])
            
            for n in range(N):
                plt.plot([Y[n][2*j],Y_hat[n][2*j]],[Y[n][2*j+1],Y_hat[n][2*j+1]],color='red')
            
            plt.legend()
            plt.show()

def discard(Y,iota):
    
    N,D = np.shape(Y)
    D1 = len(iota)
    
    if D != D1:
        print("Le nombre de dimensions ne correspond pas")
    else :
        if np.any((iota-iota**2).astype(bool)):
            print("Vecteur non-booléen")
        else :
            Y_tilde = np.transpose(np.array([Y[:,d] for d in range(D) if iota[d]]))
            return Y_tilde

def disarg(Y,iota):
    Y_v = discard(Y,iota)
    Y_u = discard(Y,1-iota)
    Y_tilde = np.concatenate([Y_v,Y_u],axis=1)
    return Y_tilde
        
def restit(Y,iota):
    
    N,U = np.shape(Y)
    D = len(iota)
    
    if np.any((iota-iota**2).astype(bool)):
        print("Vecteur non-booléen")
    else :
        iota_inv = np.array([d for d in range(D) if iota[d]])
        if len(iota_inv) != U :
            print("Le nombre de dimensions ne correspond pas")
        else :
            Y_tilde = np.zeros((N,D))
            for u in range(U):
                Y_tilde[:,iota_inv[u]] = Y[:,u]
            return Y_tilde

def rearg(Y,iota):
    
    N,D = np.shape(Y)
    D1 = len(iota)
    
    if D != D1:
        print("Le nombre de dimensions ne correspond pas")
    else :
        if np.any((iota-iota**2).astype(bool)):
            print("Vecteur non-booléen")
        else :
            Dv = np.sum(iota)
            Y_v = restit(Y[:,:Dv],iota)
            Y_u = restit(Y[:,Dv:],1-iota)
        
    return Y_u + Y_v

def da_matrix(iota):
    
    D = len(iota)
    if np.any((iota-iota**2).astype(bool)):
        print("Vecteur non-booléen")
    else:
        R = np.zeros((D,D)).astype(int)
        Dv = np.sum(iota)
        
        u = 0
        v = 0
        for d in range(D):
            if iota[d]:
                R[v][d] = 1
                v += 1
            else :
                R[Dv+u][d] = 1
                u += 1
        return R

def ra_matrix(iota):
    
    D = len(iota)
    if np.any((iota-iota**2).astype(bool)):
        print("Vecteur non-booléen")
    else:
        R = np.zeros((D,D)).astype(int)
        Dv = np.sum(iota)
        
        u = 0
        v = 0
        for d in range(D):
            if iota[d]:
                R[d][v] = 1
                v += 1
            else :
                R[d][Dv+u] = 1
                u += 1
        return R

def FS_rec1(W,Z,mu,iota):
    
    Y_tilde = Z@np.transpose(W)
    Y_hat = restit(Y_tilde, iota) + mu
    
    return Y_hat

def FS_rec2(W,Z,V,X,mu,iota):
    
    mu_X = np.mean(X,axis=0)
    Xc = X - mu_X
    
    Y_tilde = Z@np.transpose(W)
    Y_hat = restit(Y_tilde, iota) + Xc@np.transpose(V) + mu
    
    return Y_hat

def FS_mixrec1(thetas,Z,omega,iotas):
    
    N = len(omega)
    K,D = np.shape(iotas)
    Y_hat = np.zeros((N,D))
    
    for n in range(N):
        
        k = omega[n]
        W,mu,sigma2 = thetas[k]
        Y_tilde_n = np.array([W@Z[n]])
        Y_hat[n] = (restit(Y_tilde_n,iotas[k]))[0] + mu
    
    return Y_hat

def FS_mixrec2(thetas,Z,X,omega,iotas):
    
    N = len(omega)
    K,D = np.shape(iotas)
    Y_hat = np.zeros((N,D))
    
    mu_X = np.mean(X,axis=0)
    Xc = X - mu_X
    
    for n in range(N):
        
        k = omega[n]
        W,V,mu,sigma2 = thetas[k]
        Y_tilde_n = np.array([W@Z[n]])
        Y_hat[n] = (restit(Y_tilde_n,iotas[k]))[0] + V@Xc[n] + mu
    
    return Y_hat

def FS_sperec1(Wv,Wu,Zv,Zu,mu,iota):
    
    Dv,Lv = np.shape(Wv)
    Du,Lu = np.shape(Wu)
    N1,Lv1 = np.shape(Zv)
    N2,Lu1 = np.shape(Zu)
    D = len(mu)
    
    if N1 != N2 :
        print("Erreur de dimensions sur Zv et Zu")
    if D != Du+Dv:
        print("Erreur de dimensions sur Wu, Wv et mu")
    if Lv != Lv1 :
        print("Erreur de dimensions sur Wv et Zv")
    if Lu != Lu1 :
        print("Erreur de dimensions sur Wu et Zu")
        
    Y_tilde = np.concatenate([Zv@np.transpose(Wv),Zu@np.transpose(Wu)],axis=1)
    Y_hat = rearg(Y_tilde,iota) + mu
    
    return Y_hat
        
def FS_sperec2(Wv,Wu,Zv,Zu,V,X,mu,iota):
    
    Dv,Lv = np.shape(Wv)
    Du,Lu = np.shape(Wu)
    N1,Lv1 = np.shape(Zv)
    N2,Lu1 = np.shape(Zu)
    N,C = np.shape(X)
    D1,C1 = np.shape(V)
    D = len(mu)
    
    if N1 != N2 :
        print("Erreur de dimensions sur Zv et Zu")
    if D != Du+Dv:
        print("Erreur de dimensions sur Wu, Wv et mu")
    if Lv != Lv1 :
        print("Erreur de dimensions sur Wv et Zv")
    if Lu != Lu1 :
        print("Erreur de dimensions sur Wu et Zu")
    if D != D1 :
        print("Erreur de dimensions sur V et mu")
    if N != N1 :
        print("Erreur de dimensions sur Zv et X")
    if N != N2 :
        print("Erreur de dimensions sur Zu et X")
    if C != C1 :
        print("Erreur de dimensions sur V et X")
        
    Xc = X - np.mean(X,axis=0)
    Y_tilde = np.concatenate([Zv@np.transpose(Wv),Zu@np.transpose(Wu)],axis=1)
    Y_hat = rearg(Y_tilde,iota) + Xc@np.transpose(V) + mu
    
    return Y_hat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions de clustering
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def HAC_SL(X,K,tempo=False):
    
    clusters = sklclu.AgglomerativeClustering(K,linkage='single').fit(X)
    omega = clusters.labels_
    
    return omega

def HAC_CL(X,K,tempo=False):
    
    clusters = sklclu.AgglomerativeClustering(K,linkage='complete').fit(X)
    omega = clusters.labels_
    
    return omega

def HAC_AL(X,K,tempo=False):
    
    clusters = sklclu.AgglomerativeClustering(K,linkage='average').fit(X)
    omega = clusters.labels_
    
    return omega

def HAC_Ward(X,K,tempo=False):
    
    clusters = sklclu.AgglomerativeClustering(K,linkage='ward').fit(X)
    omega = clusters.labels_
    
    return omega

def K_means(X,omega,nb_steps=100,tempo=True):
    
    N,D = np.shape(X)
    occ = occurences(omega)
    K = len(occ)
    if np.any(occ==0) :
        print("K_means : Clusters vides")
    
    omega_hat = omega.copy()
    t = 0
    dist = 1
    M = np.zeros((K,D))
    
    while dist > 0 and t < nb_steps :
        
        #Recalcul des centres
        tri_X = tri(X,omega_hat,K)
        new_M = np.zeros((K,D))
        for k in range(K) :
            if len(tri_X[k]) > 0:
                new_M[k] = np.mean(tri_X[k],axis = 0)
            else :
                new_M[k] = M[k]
        M = new_M
    
        #Recalcul des clusters
        new_omega = np.zeros(N)
        for n in range(N) :
            x = X[n]
            dists_L2 = np.array([np.sum((x-M[k])**2) for k in range(K)])
            new_omega[n] = np.argmin(dists_L2)
        
        dist = np.sum((omega_hat-new_omega)**2)
        omega_hat = new_omega.astype(int)
        t += 1
    
    if tempo :
        print('t = ',t)
    
    return omega_hat

def K_means_FPC(X,K,determ=True,nb_steps=100,tempo=True):
    
    N = len(X)
    if N < K :
        print("N < K")
    else :
        
        #Initialisation
        
        if determ :
            Pdm = dissim_L2(X)
            ind_max = np.argmax(Pdm)
            n0 = int(ind_max/N)
            n1 = ind_max%N
            
            M = np.array([X[n0],X[n1]])
            
            for k in range(2,K):
            
                dists_CP = np.array([Dist_CP(x,M) for x in X])
                n_k = np.argmax(dists_CP)
                
                M_list = list(M)
                M_list.append(X[n_k])
                M = np.array(M_list)
        
        else :
            n0 = rd.choice(N)
            M = np.array([X[n0]])
            
            for k in range(1,K):
                
                dists_CP = np.array([Dist_CP(x,M) for x in X])
                
                tot_dist = np.sum(dists_CP)
                probas = dists_CP/tot_dist
                n_k = rd.choice(N,p=probas)
                
                M_list = list(M)
                M_list.append(X[n_k])
                M = np.array(M_list)
            
        omega = (-np.ones(N)).astype(int)
        for n in range(N):
            x = X[n]
            dists_L2 = np.array([np.sum((x-M[k])**2) for k in range(K)])
            omega[n] = int(np.argmin(dists_L2))
        
        #K-means
        return K_means(X,omega,nb_steps,tempo)

def K_medoids(X,omega,nb_steps=100,tempo=True):
    
    N,D = np.shape(X)
    occ = occurences(omega)
    K = len(occ)
    if np.any(occ==0) :
        print("K_means : Clusters vides")
    
    omega_hat = omega.copy()
    t = 0
    dist = 1
    M = np.zeros((K,D))
    
    while dist > 0 and t < nb_steps :
        
        #Recalcul des médoïdes
        tri_X = tri(X,omega_hat,K)
        for k in range(K):
            dist_sums = np.array([np.sum(np.array([np.sum((x-y)**2) for x in tri_X[k]])) for y in tri_X[k]])
            n_k = np.argmin(dist_sums)
            M[k] = X[n_k]
        
        #Recalcul des clusters
        new_omega = np.zeros(N)
        for n in range(N) :
            x = X[n]
            dists_L2 = np.array([np.sum((x-M[k])**2) for k in range(K)])
            new_omega[n] = np.argmin(dists_L2)
        
        dist = np.sum((omega_hat-new_omega)**2)
        omega_hat = new_omega.astype(int)
        t += 1
    
    if tempo :
        print('t = ',t)
    
    return omega_hat

#Vecteurs propres du Laplacien du graphe

def Lapras(X,K,nb_steps=100,tempo=True):
    
    N,D = np.shape(X)
    
    #Initialisation
    Pdm = dissim_L2(X)
    alpha = np.mean(Pdm)
    Psm = np.exp(-N/(N-1)/alpha*Pdm)
    Lokh = Lap(Psm)

    #EVP
    SpL,P = nla.eig(Lokh)
    P2 = normalize(P)
    ordre = np.sort(SpL)
    inlist = [k for k in range(N) if SpL[k] in ordre[:K]]
    tU = np.array([P2[:,k] for k in inlist])
    T = np.transpose(normalize(tU))
    omega = K_means_FPC(T,K,nb_steps=nb_steps,tempo=tempo)
    
    return omega

#K optimal
def K_opt(X,alpha=None,beta=None,detail=False):
    
    N,D = np.shape(X)    
    Pdm = dissim_L2(X)
    
    if type(alpha) == type(None):
        alpha = 0.5/np.mean(Pdm)
    
    if type(beta) == type(None):
        beta = 0.5
    
    Psm = np.exp(-alpha*Pdm)
    Lokh = Lap(Psm)
    SpL,P = nla.eig(Lokh)
    
    log_ordre = np.log(np.sort(SpL)[1:])
    K_star = L_knee(log_ordre,beta) + 1
    
    if detail:
        
        plt.figure()
        plt.step(np.arange(1,N),log_ordre)
        plt.plot([1,K_star,K_star],[log_ordre[K_star-1],log_ordre[K_star-1],log_ordre[0]],'--',label='$K_{star}$')
        plt.title('Log du spectre de la Laplacienne de la matrice de similarité')
        plt.legend()
        plt.figure()
    
    return K_star

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions de MFA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def MFA_obs1(Y,L,K=None,omega=None,fun=Lapras,nb_steps=100,orthog=True,tempo=True):
    
    N,D = np.shape(Y)
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = K_opt(Y)
                
        omega_hat = fun(Y,K,tempo=tempo)
        omega_hat = K_means(Y,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    reclus = [[n for n in range(N) if omega_hat[n] == k] for k in range(K)]
    tri_Y = tri(Y,omega_hat)
    thetas_hat = [[] for k in range(K)]
    Z_hat = np.zeros((N,L))
    
    for k in range(K):
        
        y = tri_Y[k]
        card_k = len(y)
        mu_k = np.mean(y,axis=0)
        W_k_hat, Z_k_hat, sigma2_k_hat = PPCA(y,L,orthog)
        
        thetas_hat[k] = [W_k_hat, mu_k, sigma2_k_hat]
        for j in range(card_k):
            Z_hat[reclus[k][j]] = Z_k_hat[j]
        
    return thetas_hat, Z_hat, omega_hat

def MFA_obs2(Y,X,L,K=None,omega=None,fun=Lapras,nb_steps=100,err=0.0,orthog=True,tempo=True):
    
    N,D = np.shape(Y)
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = K_opt(Y)
                
        omega_hat = fun(Y,K,tempo=tempo)
        omega_hat = K_means(Y,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    reclus = [[n for n in range(N) if omega_hat[n] == k] for k in range(K)]
    tri_Y = tri(Y,omega_hat)
    tri_X = tri(X,omega_hat)
    thetas_hat = [[] for k in range(K)]
    Z_hat = np.zeros((N,L))
    
    for k in range(K):
        
        y = tri_Y[k]
        x = tri_X[k]
        card_k = len(y)
        mu_k = np.mean(y,axis=0)
        W_k_hat, Z_k_hat, V_k_hat, Sigma_k_hat, sigma2_k_hat = ML_RCA_2(y,x,L,nb_steps=nb_steps,err=err,tempo=tempo)
        
        thetas_hat[k] = [W_k_hat, V_k_hat, mu_k, sigma2_k_hat]
        for j in range(card_k):
            Z_hat[reclus[k][j]] = Z_k_hat[j]
        
    return thetas_hat, Z_hat, omega_hat

def MFA_lat1(Y,L=None,K=None,omega=None,fun=Lapras,nb_steps=100,orthog=True,tempo=True):
    
    N,D = np.shape(Y)
    W_hat, Z_hat, sigma2_hat = PPCA(Y,L,orthog)
    mu_hat = np.mean(Y,axis=0)
    eta_hat = W_hat,mu_hat,sigma2_hat
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = K_opt(Z_hat)
                
        omega_hat = fun(Z_hat,K,tempo=tempo)
        omega_hat = K_means(Z_hat,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    tri_Z = tri(Z_hat,omega_hat)
    thetas_hat = []
    
    for k in range(K):
        z = tri_Z[k]
        thetas_k = MLE_Gauss(z)
        thetas_hat.append(thetas_k)
        
    return eta_hat, thetas_hat, Z_hat, omega_hat

def MFA_lat2(Y,X,L,K=None,omega=None,fun=Lapras,nb_steps=100,err=0.0,orthog=True,tempo=True):
    
    N,D = np.shape(Y)
    N1,C = np.shape(X)
    W_hat, Z_hat, V_hat, Sigma_hat, sigma2_hat = ML_RCA_2(Y,X,L,nb_steps=nb_steps,err=err,tempo=tempo)
    mu_hat = np.mean(Y,axis=0)
    eta_hat = W_hat,V_hat,mu_hat,sigma2_hat
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = K_opt(Z_hat)
                
        omega_hat = fun(Z_hat,K,tempo=tempo)
        omega_hat = K_means(Z_hat,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    tri_Z = tri(Z_hat,omega_hat)
    thetas_hat = []
    
    for k in range(K):
        z = tri_Z[k]
        thetas_k = MLE_Gauss(z)
        thetas_hat.append(thetas_k)
        
    return eta_hat, thetas_hat, Z_hat, omega_hat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions pour simuler des données bruitées
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sim_noisy_param_1(D,L,U,m1=0.0,m2=0.0,m3=0.0,s1=1.0,s2=1.0,s3=1.0,disp=False,orthog=True):
    W = rd.normal(m1,s1**2,(U,L))
    mu = rd.normal(m2,s2**2,D)
    sigma2 = rd.normal(m3,s3**2)**2
    
    if orthog:
        W = orthogonalize(W)
    
    if disp:
        print('$W = $', W)
        print('$\mu = $', mu)
        print('$\sigma^2 = $', sigma2)
        
    return W, mu, sigma2

def sim_noisy_data_1(W,mu,sigma2,N,disp=False):
    
    U,L = np.shape(W)
    D = len(mu)
    Z = rd.normal(0,1,(N,L))
    Y_prov = Z@np.transpose(W)
    iota_prov = np.concatenate([np.ones(U), np.zeros(D-U)]).astype(int)
    
    noise = rd.normal(0,sigma2,(N,D))
    iota = rd.perumtation(iota_prov)
    Y = restit(Y_prov,iota) + noise + mu
    
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\iota = $', iota)
    return Z, Y, iota

def sim_noisy_param_2(D,L,C,U,m1=0.0,m2=0.0,m3=0.0,m4=0.0,s1=1.0,s2=1.0,s3=1.0,s4=1.0,disp=False,orthog=True):
    
    W = rd.normal(m1,s1**2,(U,L))
    V = rd.normal(m2,s2**2,(D,C))
    mu = rd.normal(m3,s3**2,D)
    sigma2 = rd.normal(m4,s4**2)**2
    
    if orthog:
        W = orthogonalize(W)
        V = orthogonalize(V)
    
    if disp:
        print('$W = $', W)
        print('$V = $', V)
        print('$\mu = $', mu)
        print('$\sigma^2 = $', sigma2)
        
    return W, V, mu, sigma2

def sim_noisy_data_2(W,V,mu,sigma2,N,Sigma_X=None,disp=False):
    
    U,L = np.shape(W)
    D,C = np.shape(V)
        
    Z = rd.normal(0,1,(N,L))
    
    if type(Sigma_X) == type(None) :
        X = rd.normal(0,1,(N,C))
    else :
        X = rd.multivariate_normal(np.zeros(C),Sigma_X,N)
    
    Y_prov = Z@np.transpose(W)
    iota_prov = np.concatenate([np.ones(U), np.zeros(D-U)]).astype(int)
    
    noise = rd.normal(0,sigma2,(N,D))
    iota = rd.perumtation(iota_prov)
    Y = restit(Y_prov,iota) + X@np.transpose(V) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$X = $', X)
        print('$\iota = $', iota)
        
    return Z, Y, X, iota

def sim_noisy_param_obsmix_1(K,D,L,U,m1=0.0,m2=0.0,m3=0.0,s1=1.0,s2=1.0,s3=1.0,disp=False,orthog=True):
    thetas = [sim_noisy_param_1(D,L,U,m1,m2,m3,s1,s2,s3,disp,orthog) for k in range(K)]
    return thetas

def sim_noisy_param_obsmix_2(K,D,L,C,U,m1=0.0,m2=0.0,m3=0.0,m4=0.0,s1=1.0,s2=1.0,s3=1.0,s4=1.0,disp=False,orthog=True):
    thetas = [sim_noisy_param_2(D,L,C,U,m1,m2,m3,m4,s1,s2,s3,s4,disp,orthog) for k in range(K)]
    return thetas

def sim_noisy_param_latmix_1(K,D,L,U,m2=0.0,m3=0.0,s2=1.0,s3=1.0,m_glob1=0.0,m_glob2=0.0,m_glob3=0.0,s_glob1=1.0,s_glob2=1.0,s_glob3=1.0,disp=False,orthog=True):
    eta = sim_noisy_param_1(D,L,U,m_glob1,m_glob2,m_glob3,s_glob1,s_glob2,s_glob3,disp,orthog)
    thetas = [sim_param(L,L,0.0,m2,m3,1.0,s2,s3,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def sim_noisy_param_latmix_2(K,D,L,C,U,m2=0.0,m3=0.0,s2=1.0,s3=1.0,m_glob1=0.0,m_glob2=0.0,m_glob3=0.0,m_glob4=0.0,s_glob1=1.0,s_glob2=1.0,s_glob3=1.0,s_glob4=1.0,disp=False,orthog=True):
    eta = sim_noisy_param_2(D,L,C,U,m_glob1,m_glob2,m_glob3,m_glob4,s_glob1,s_glob2,s_glob3,s_glob4,disp,orthog)
    thetas = [sim_param(L,L,0.0,m2,m3,1.0,s2,s3,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def sim_noisy_data_obsmix_1(thetas,N,N_min=0,disp=False):
    
    K = len(thetas)
    U,L = np.shape(thetas[0][0])
    D = len(thetas[0][1])
    
    omega = sim_omega(N,K,int(max(L+1,N_min)))
    K = int(max(omega)) + 1
    
    Z = rd.normal(0,1,(N,L))
    iotas_prov = np.concatenate([np.ones((K,U)),np.zeros((K,D-U))],axis=1).astype(int)
    iotas = np.array([rd.permutation(iota) for iota in iotas_prov])
    
    Y = FS_mixrec1(thetas,Z,omega,iotas) + np.array([rd.normal(0,thetas[omega[n]][2],D) for n in range(N)])
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\omega = $', omega)
        print('$\iotas = $', iotas)
        
    return Z, omega, Y, iotas

def sim_noisy_data_obsmix_2(thetas,N,N_min=0,Sigma_X=None,disp=False):
    
    K = len(thetas)
    U,L = np.shape(thetas[0][0])
    D,C = np.shape(thetas[0][1])
    
    omega = sim_omega(N,K,int(max(L+1,C+1,N_min)))
    K = int(max(omega)) + 1
    
    Z = rd.normal(0,1,(N,L))
    if type(Sigma_X) == type(None) :
        X = rd.normal(0,1,(N,C))
    else :
        X = rd.multivariate_normal(np.zeros(C),Sigma_X,N)
    
    iotas_prov = np.concatenate([np.ones((K,U)),np.zeros((K,D-U))],axis=1).astype(int)
    iotas = np.array([rd.permutation(iota) for iota in iotas_prov])
    
    Y = FS_mixrec2(thetas,Z,X,omega,iotas) + np.array([rd.normal(0,thetas[omega[n]][3],D) for n in range(N)])
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
        print('$\iotas = $', iotas)
        
    return Z, omega, Y, X, iotas

def sim_noisy_data_latmix_1(eta,thetas,N,N_min=2,disp=False):
    
    W, mu, sigma2 = eta
    U,L = np.shape(W)
    D = len(mu)
    K = len(thetas)
    
    omega = sim_omega(N,K,N_min)
    K = int(max(omega)) + 1
    
    Z_orig = rd.normal(0,1,(N,L))
    Z = np.zeros((N,L))
    for n in range(N):
        k = omega[n]
        mu_k,sigma2_k = thetas[k]
        Z[n] = np.sqrt(sigma2_k)*Z_orig[n] + mu_k
        
    Y_prov = Z@np.transpose(W)
    iota_prov = np.concatenate([np.ones(U), np.zeros(D-U)]).astype(int)
    
    noise = rd.normal(0,sigma2,(N,D))
    iota = rd.permutation(iota_prov)
    
    Y = restit(Y_prov,iota) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\omega = $', omega)
        print('$\iota = $', iota)
    
    return Z, omega, Y, iota

def sim_noisy_data_latmix_2(eta,thetas,N,N_min=2,Sigma_X=None,disp=False):
    
    W, V, mu, sigma2 = eta
    U,L = np.shape(W)
    D,C = np.shape(V)
    K = len(thetas)
    
    omega = sim_omega(N,K,N_min)
    K = int(max(omega)) + 1
    
    Z_orig = rd.normal(0,1,(N,L))
    Z = np.zeros((N,L))
    for n in range(N):
        k = omega[n]
        mu_k,sigma2_k = thetas[k]
        Z[n] = np.sqrt(sigma2_k)*Z_orig[n] + mu_k
        
    if type(Sigma_X) == type(None) :
        X = rd.normal(0,1,(N,C))
    else :
        X = rd.multivariate_normal(np.zeros(C),Sigma_X,N)
        
    noise = rd.normal(0,sigma2,(N,D))
    Y_prov = np.concatenate([Z@np.transpose(W),np.zeros((N,D-U))],axis=1)
    iota_prov = np.concatenate([np.ones((1,U)),np.zeros((1,D-U))],axis=1).astype(int)
    iota = rd.permutation(iota_prov)
    
    Y = restit(Y_prov,iota) + X@np.transpose(V) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
        print('$\iota = $', iota)
    
    return Z, omega, Y, X, iota

def sim_noisy_param_spemix_1(K,Dv,Du,Lv,Lu,m1u=0.0,m1v=0.0,m2u=0.0,m2v=0.0,m3u=0.0,m3v=0.0,m3t=0.0,s1u=1.0,s1v=1.0,s2u=1.0,s2v=1.0,s3u=1.0,s3v=1.0,s3t=1.0,disp=False,orthog=True):
    Wu = rd.normal(m1u,s1u**2,(Du,Lu))
    Wv = rd.normal(m1v,s1v**2,(Dv,Lv))
    
    D = Du+Dv
    mu = rd.normal(m2u,s2u**2,D)
    sigma2 = rd.normal(m3u,s3u**2)**2
    tau2 = rd.normal(m3t,s3t**2)**2
    
    thetas = [[] for k in range(K)]    
    for k in range(K):
        mu_k = rd.normal(m2v,s2v**2,Lv)
        sigma2_k = rd.normal(m3v,s3v**2)**2
        thetas[k] = [mu_k,sigma2_k]    
    
    if orthog:
        Wu = orthogonalize(Wu)
        Wv = orthogonalize(Wv)
    
    eta = [Wv,Wu,mu,sigma2,tau2]
    
    if disp:
        print("eta =",eta)
        print("thetas =",thetas)
        
    return eta, thetas

def sim_noisy_param_spemix_2(K,Dv,Du,Lv,Lu,C,m1u=0.0,m1v=0.0,m2=0.0,m3u=0.0,m3v=0.0,m4u=0.0,m4v=0.0,m4t=0.0,s1u=1.0,s1v=1.0,s2=1.0,s3u=1.0,s3v=1.0,s4u=1.0,s4v=1.0,s4t=1.0,disp=False,orthog=True):
    Wu = rd.normal(m1u,s1u**2,(Du,Lu))
    Wv = rd.normal(m1v,s1v**2,(Dv,Lv))
    D = Du+Dv
    V = rd.normal(m2,s2**2,(D,C))
    
    
    mu = rd.normal(m3u,s3u**2,D)
    sigma2 = rd.normal(m4u,s4u**2)**2
    tau2 = rd.normal(m4t,s4t**2)**2
    
    thetas = [[] for k in range(K)]    
    for k in range(K):
        mu_k = rd.normal(m3v,s3v**2,Lv)
        sigma2_k = rd.normal(m4v,s4v**2)**2
        thetas[k] = [mu_k,sigma2_k]    
    
    if orthog:
        Wu = orthogonalize(Wu)
        Wv = orthogonalize(Wv)
        V = orthogonalize(V)
    
    eta = [Wv,Wu,V,mu,sigma2,tau2]
    
    if disp:
        print("eta =",eta)
        print("thetas =",thetas)
        
    return eta, thetas

def sim_noisy_data_spemix_1(eta,thetas,N,N_min=2,disp=False):
    
    Wv, Wu, mu, sigma2, tau2 = eta
    Du,Lu = np.shape(Wu)
    Dv,Lv = np.shape(Wv)
    D = len(mu)
    K = len(thetas)
    
    if D != Du + Dv:
        print("Erreur de dimensions sur Wu, Wv et mu")
    
    omega = sim_omega(N,K,N_min)
    K = int(max(omega)) + 1
    
    Z_orig = rd.normal(0,1,(N,Lv))
    Zv = np.zeros((N,Lv))
    for n in range(N):
        k = omega[n]
        mu_k,sigma2_k = thetas[k]
        Zv[n] = np.sqrt(sigma2_k)*Z_orig[n] + mu_k
    
    Zu = rd.normal(0,sigma2,(N,Lu))
    
    Y_prov = np.concatenate([Zv@np.transpose(Wv),Zu@np.transpose(Wu)],axis=1)
    iota_prov = np.concatenate([np.ones(Dv), np.zeros(Du)]).astype(int)
    
    noise = rd.normal(0,tau2,(N,D))
    iota = rd.permutation(iota_prov)
    
    Y = rearg(Y_prov,iota) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\omega = $', omega)
        print('$\iota = $', iota)
    
    return Zv, Zu, omega, Y, iota

def sim_noisy_data_spemix_2(eta,thetas,N,N_min=2,Sigma_X=None,disp=False):
    
    Wv, Wu, V, mu, sigma2, tau2 = eta
    Du,Lu = np.shape(Wu)
    Dv,Lv = np.shape(Wv)
    D,C = np.shape(V)
    K = len(thetas)
    
    if D != Du + Dv:
        print("Erreur de dimensions sur Wu, Wv et V")
    
    omega = sim_omega(N,K,N_min)
    K = int(max(omega)) + 1
    
    Z_orig = rd.normal(0,1,(N,Lv))
    Zv = np.zeros((N,Lv))
    for n in range(N):
        k = omega[n]
        mu_k,sigma2_k = thetas[k]
        Zv[n] = np.sqrt(sigma2_k)*Z_orig[n] + mu_k
    
    Zu = rd.normal(0,sigma2,(N,Lu))
    
    Y_prov = np.concatenate([Zv@np.transpose(Wv),Zu@np.transpose(Wu)],axis=1)
    iota_prov = np.concatenate([np.ones(Dv), np.zeros(Du)]).astype(int)
    
    noise = rd.normal(0,tau2,(N,D))
    iota = rd.permutation(iota_prov)
    
    if type(Sigma_X) == type(None) :
        X = rd.normal(0,1,(N,C))
    else :
        X = rd.multivariate_normal(np.zeros(C),Sigma_X,N)
    
    Y = rearg(Y_prov,iota) + X@np.transpose(V) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
        print('$\iota = $', iota)
    
    return Zv, Zu, omega, Y, X, iota

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions utiles pour la sélection de variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def cor_emp(Y,Z):
    
    N1,D = np.shape(Y)
    N2,L = np.shape(Z)
    
    if N1 != N2:
        print("Y et Z sont de tailles différentes")
    else :
        N = N1
        
        mu_Y = np.mean(Y,axis=0)
        vars_Y = np.var(Y,axis=0)
        Yc = Y - mu_Y
        
        mu_Z = np.mean(Z,axis=0)
        vars_Z = np.var(Z,axis=0)
        Zc = Z - mu_Z
        
        Cor = 1/N *(np.diag(1/np.sqrt(vars_Z)) @ np.transpose(Zc) @ Yc @ np.diag(1/np.sqrt(vars_Y)))
        
        return Cor

def iotate(Y,Z,U=None,dist='2',detail=False):
    
    N1,D = np.shape(Y)
    N2,L = np.shape(Z)
    
    if N1 != N2:
        print("Y et Z sont de tailles différentes")
    else :
        N = N1
    
        Cor = cor_emp(Y,Z)
        
        if dist == 'inf':
            contrib = np.max(np.abs(Cor),axis=0)
        if dist == '2':
            contrib = np.sqrt(np.sum(Cor**2,axis=0))
            
        ordre = np.sort(contrib)
        
        if type(U) == type(None):
            U = U_opt(ordre,L,detail)
        
        brink = ordre[D-U]
        iota_hat = (contrib>=brink).astype(int)
        
        return iota_hat
    
def U_opt(contrib,U_min=1,detail=False):
    
    D = len(contrib)
    
    if U_min < 1:
        U_min = 1
    
    if D <= U_min:
        print("D inférieur ou égal à U_min")
    else :
        ordre = np.sort(contrib)
        diff_ordre = np.concatenate([[0],ordre[1:D-U_min+1] - ordre[:D-U_min]])
        U_star = D - int(np.argmax(diff_ordre))
        
        if detail:
            
            plt.figure()
            plt.step(np.arange(0,D),ordre,label='Contribution aux axes principaux')
            plt.plot([D-U_star-1,D-U_star-1],[ordre[D-U_star-1],ordre[D-U_star]])
            plt.plot([D-U_star-1,D-U_star-1],[ordre[D-U_star-1],0],'--',label='$U_{star}$')
            plt.legend()
            plt.show()
        
        return U_star

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions de FS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def FS_PPCA(Y,L=None,orthog=True,U=None):
    
    W_hat,Z_hat,sigma2_hat = PPCA(Y,L,orthog)
    iota_hat = iotate(Y,Z_hat,U)
    
    U,L = np.shape(W_hat)
    N,D = np.shape(Y)
    Y_tilde = discard(Y,iota_hat)
    
    W_hac,Z_hac,sigma2_hac = PPCA(Y_tilde,L,orthog)
    
    return W_hac,Z_hac,sigma2_hac,iota_hat

def FS_RCA(Y,X,L,V=None,Sigma=None,nb_steps=100,err=0.0,tempo=True,U=None,orthog=True):
    
    W_hat, Z_hat, V_hat, Sigma_hat, sigma2_hat = ML_RCA_2(Y,X,L,V,Sigma,nb_steps,err,tempo)
    mu_X = np.mean(X,axis=0)
    Xc = X - mu_X
    
    iota_hat = iotate(Y-Xc@np.transpose(V_hat),Z_hat,U)
    D,C = np.shape(V_hat)
    U,L = np.shape(W_hat)
    N = len(Y)
    Y_tilde = discard(Y-Xc@np.transpose(V_hat),iota_hat)
    
    W_hac,Z_hac,sigma2_hac = PPCA(Y_tilde,L,orthog)
    
    return W_hac, Z_hac, V_hat, Sigma_hat, sigma2_hac, iota_hat

def FS_MFA_lat1(Y,L=None,K=None,omega=None,fun=Lapras,nb_steps=100,orthog=True,tempo=True,U=None):
    
    N,D = np.shape(Y)
    W_hat,Z_hat,sigma2_hat = PPCA(Y,L,orthog)
    D,L = np.shape(W_hat)
    
    mu_hat = np.mean(Y,axis=0)
    
    iota_hat = iotate(Y,Z_hat,U)
    
    U = np.sum(iota_hat)
    Y_tilde = discard(Y,iota_hat)
    
    W_hac,Z_hac,sigma2_hac = PPCA(Y_tilde,L,orthog)
        
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = K_opt(Z_hac)
                
        omega_hat = fun(Z_hac,K,tempo=tempo)
        omega_hat = K_means(Z_hac,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    tri_Z = tri(Z_hac,omega_hat)
    thetas_hac = []
    
    for k in range(K):
        z = tri_Z[k]
        thetas_k = MLE_Gauss(z)
        thetas_hac.append(thetas_k)
    
    Y_hat = FS_rec2(W_hac,Z_hac,mu_hat,iota_hat)
    sigma2_hac = np.mean((Y-Y_hat)**2)
    eta_hac = W_hac,mu_hat,sigma2_hac
    
    return eta_hac, thetas_hac, Z_hac, omega_hat, iota_hat

def FS_MFA_lat2(Y,X,L,K=None,omega=None,fun=Lapras,nb_steps=100,err=0.0,orthog=True,tempo=True,U=None):
    
    N,D = np.shape(Y)
    N1,C = np.shape(X)
    W_hat, Z_hat, V_hat, Sigma_hat, sigma2_hat = ML_RCA_2(Y,X,L,nb_steps=nb_steps,err=err,tempo=tempo)
    
    mu_hat = np.mean(Y,axis=0)
    mu_X = np.mean(X,axis=0)
    Xc = X - mu_X
    
    iota_hat = iotate(Y-Xc@np.transpose(V_hat),Z_hat,U)
    
    U = np.sum(iota_hat)
    Y_tilde = discard(Y-Xc@np.transpose(V_hat),iota_hat)
    
    W_hac,Z_hac,sigma2_hac = PPCA(Y_tilde,L,orthog)
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = K_opt(Z_hac)
                
        omega_hat = fun(Z_hac,K,tempo=tempo)
        omega_hat = K_means(Z_hac,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    tri_Z = tri(Z_hac,omega_hat)
    thetas_hat = []
    
    for k in range(K):
        z = tri_Z[k]
        thetas_k = MLE_Gauss(z)
        thetas_hat.append(thetas_k)
    
    Y_hat = FS_rec2(W_hac,Z_hac,V_hat,X,mu_hat,iota_hat)
    sigma2_hac = np.mean((Y-Y_hat)**2)
    eta_hac = W_hac,V_hat,mu_hat,sigma2_hac
        
    return eta_hac, thetas_hat, Z_hac, omega_hat, iota_hat

def FS_MFA_obs1(Y,L,K=None,omega=None,fun=Lapras,nb_steps=100,orthog=True,tempo=True,U=None):
    
    N,D = np.shape(Y)
    thetas_hat, Z_hat, omega_hat = MFA_obs1(Y,L,K,omega,fun,nb_steps,orthog,tempo)
    
    K = len(thetas_hat)
    tri_Y = tri(Y,omega_hat)
    tri_Z = tri(Z_hat,omega_hat)
    
    iotas_hat = np.zeros((K,D)).astype(int)
    reclus = tri((np.arange(N)).astype(int),omega_hat)
    Z_hac = np.zeros((N,L))
    thetas_hac = [[] for k in range(K)]
    
    for k in range(K):
        
        Y_k = tri_Y[k]
        Z_k = tri_Z[k]
        card_k = len(Y_k)
        
        mu_k_hac = np.mean(Y_k,axis=0)
        iota_k_hat = iotate(Y_k,Z_k,U)
        iotas_hat[k] = iota_k_hat
        U = np.sum(iota_k_hat)
        
        Y_k_tilde = discard(Y_k,iota_k_hat)
        W_k_hac, Z_k_hac, sigma2_k_hac = PPCA(Y_k_tilde,L,orthog)
        
        for j in range(card_k):
            Z_hac[reclus[k][j]] = Z_k_hac[j]
        
        Y_k_hac = FS_rec1(W_k_hac,Z_k_hac,mu_k_hac,iota_k_hat)
        sigma2_k_hac = np.mean((Y_k-Y_k_hac)**2)
            
        thetas_hac[k] = [W_k_hac,mu_k_hac,sigma2_k_hac]
        
    return thetas_hac, Z_hac, omega_hat, iotas_hat

def FS_MFA_obs2(Y,X,L,K=None,omega=None,fun=Lapras,nb_steps=100,err=0.0,orthog=True,tempo=True,U=None):
    
    N,D = np.shape(Y)
    thetas_hat, Z_hat, omega_hat = MFA_obs2(Y,X,L,K,omega,fun,nb_steps,orthog,tempo)
    
    K = len(thetas_hat)
    tri_Y = tri(Y,omega_hat)
    tri_Z = tri(Z_hat,omega_hat)
    tri_X = tri(X,omega_hat)
    
    iotas_hat = np.zeros((K,D)).astype(int)
    reclus = tri((np.arange(N)).astype(int),omega_hat)
    Z_hac = np.zeros((N,L))
    thetas_hac = [[] for k in range(K)]
    
    for k in range(K):
        
        Y_k = tri_Y[k]
        Z_k = tri_Z[k]
        X_k = tri_X[k]
        card_k = len(Y_k)
        W_k_hat, V_k_hat, mu_k_hat, sigma_2_k_hat = thetas_hat[k]
        
        mu_k_hac = np.mean(Y_k,axis=0)
        mu_k_X = np.mean(X_k,axis=0)
        Xc_k = X_k - mu_k_X
        
        iota_k_hat = iotate(Y_k-Xc_k@np.transpose(V_k_hat),Z_k,U)
        iotas_hat[k] = iota_k_hat
        U = np.sum(iota_k_hat)
        
        Y_k_tilde = discard(Y_k-Xc_k@np.transpose(V_k_hat),iota_k_hat)
        W_k_hac, Z_k_hac, sigma2_k_hac = PPCA(Y_k_tilde,L,orthog)
        
        for j in range(card_k):
            Z_hac[reclus[k][j]] = Z_k_hac[j]
        
        Y_k_hac = FS_rec2(W_k_hac,Z_k_hac,V_k_hat,Xc_k,mu_k_hac,iota_k_hat)
        sigma2_k_hac = np.mean((Y_k-Y_k_hac)**2)
            
        thetas_hac[k] = [W_k_hac,V_k_hat,mu_k_hac,sigma2_k_hac]
        
    return thetas_hac, Z_hac, omega_hat, iotas_hat

def FS_MFA_spe1(Y,L=None,K=None,omega=None,fun=Lapras,orthog=True,tempo=True,latent=True,Dv=None,Lv=None,Lu=None):
    
    N,D = np.shape(Y)    
    mu_hat = np.mean(Y,axis=0)
    
    if type(Dv) != type(None) and Dv > D:
        print("Dv > D")
        Dv = None
    
    if type(omega) == type(None):
        if latent :
            W_hat, Z_hat, tau2_hat = PPCA(Y,L,orthog)
            if type(K) == type(None):
                K = K_opt(Z_hat)
            omega_hat = fun(Z_hat,K,tempo=tempo)
        else :
            if type(K) == type(None):
                K = K_opt(Y)
            omega_hat = fun(Y,K,tempo=tempo)
    else :
        omega_hat = omega
        
    O_hat = matrixage(omega_hat)
    iota_hat = iotate(Y,O_hat,Dv)
    
    Dv = np.sum(iota_hat)
    Du = D-Dv
    
    if type(Lv) != type(None) and Lv >= Dv:
        print("Lv >= Dv")
        Lv = None
        
    if type(Lu) != type(None) and Lu >= Du:
        print("Lu >= Du")
        Lu = None
    
    Y_tilde = disarg(Y,iota_hat)
    Yv = Y_tilde[:,:Dv]
    Yu = Y_tilde[:,Dv:]
    
    etav_hat, thetas_hat, Zv_hat, omega_hat = MFA_lat1(Yv,L=Lv,K=K,omega=omega_hat,orthog=orthog,tempo=tempo)
    Wv_hat = etav_hat [0]
    Wu_hat, Zu_hat, sig2_uosef = PPCA(Yu,Lu,orthog)
    mu_uosef,sigma2_hat = MLE_Gauss(Zu_hat)
    
    Y_hat = FS_sperec1(Wv_hat,Wu_hat,Zv_hat,Zu_hat,mu_hat,iota_hat)
    tau2_hat = np.mean((Y-Y_hat)**2)
    eta_hac = [Wv_hat,Wu_hat,mu_hat,sigma2_hat,tau2_hat]
    
    return eta_hac, thetas_hat, Zv_hat, Zu_hat, omega_hat, iota_hat

def FS_MFA_spe2(Y,X,L,K=None,omega=None,fun=Lapras,V=None,Sigma=None,nb_steps=100,err=0.0,orthog=True,tempo=True,latent=True,Dv=None,Lv=None,Lu=None):
    
    N,D = np.shape(Y)
    N1,C = np.shape(X)
    mu_hat = np.mean(Y,axis=0)
    Xc = X - np.mean(X,axis=0)
    
    if N != N1:
        print("Erreur de dimension sur Y et X")
    
    if type(Dv) != type(None) and Dv > D:
        print("Dv > D")
        Dv = None
    
    W_hat, Z_hat, V_hat, Sigma_hat, tau2_hat = ML_RCA_2(Y,X,L,V,Sigma,nb_steps,err,tempo)
    
    if type(omega) == type(None):
        if latent :
            if type(K) == type(None):
                K = K_opt(Z_hat)
            omega_hat = fun(Z_hat,K,tempo=tempo)
        else :
            if type(K) == type(None):
                K = K_opt(Y-Xc@np.transpose(V_hat))
            omega_hat = fun(Y-Xc@np.transpose(V_hat),K,tempo=tempo)
    else :
        omega_hat = omega
        
    O_hat = matrixage(omega_hat)
    iota_hat = iotate(Y-Xc@np.transpose(V_hat),O_hat,Dv)
    
    Dv = np.sum(iota_hat)
    Du = D-Dv
    
    if type(Lv) != type(None) and Lv >= Dv:
        print("Lv >= Dv")
        Lv = None
        
    if type(Lu) != type(None) and Lu >= Du:
        print("Lu >= Du")
        Lu = None
    
    Y_tilde = disarg(Y-Xc@np.transpose(V_hat),iota_hat)
    Yv = Y_tilde[:,:Dv]
    Yu = Y_tilde[:,Dv:]
    
    etav_hat, thetas_hat, Zv_hat, omega_hat = MFA_lat1(Yv,L=Lv,K=K,omega=omega_hat,orthog=orthog,tempo=tempo)
    Wv_hat = etav_hat[0]
    Wu_hat, Zu_hat, sig2_uosef = PPCA(Yu,Lu,orthog)
    mu_uosef,sigma2_hat = MLE_Gauss(Zu_hat)
    
    Y_hat = FS_sperec2(Wv_hat,Wu_hat,Zv_hat,Zu_hat,V_hat,X,mu_hat,iota_hat)
    tau2_hat = np.mean((Y-Y_hat)**2)
    eta_hac = [Wv_hat,Wu_hat,V_hat,mu_hat,sigma2_hat,tau2_hat]
    
    return eta_hac, thetas_hat, Zv_hat, Zu_hat, omega_hat, iota_hat