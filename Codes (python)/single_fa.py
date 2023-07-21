'''===========================================
    Fonctions d'estimation pour modèles simples
    ===========================================
    
    ================================== ========================================================================
    Contient les fonctions suivantes :
    ---------------------------------- ------------------------------------------------------------------------
    L_opt                              Nombre optimal de dimensions latentes, estimé par la "méthode du saut"
    L_opt_2                            Nombre optimal de dimensions latentes, estimé par la "méthode du coude".
    L_opt_3                            Nombre optimal de dimensions latentes, estimé par la "méthode du genou".
    PCA                                Analyse en Composante Principale (Adapté pour le modèle (M.1)).
    bruit                              Estimation de la variance du bruit à partir d'une matrice de covariance.
    PPCA_EM                            Analyse en Composante Principale Probabiliste, algorithme E-M
                                       (Adapté pour le modèle (M.1)).
    PPCA                               Analyse en Composante Principale Probabiliste, algorithme direct
                                       (Adapté pour le modèle (M.1)).
    ML_RCA                             Analyse en Composante Résiduelle, méthode itérative
                                       (Adapté pour le modèle (M.2)).
    MLE_Gauss                          Estimation de la moyenne et de la variance d'un N-échantillon Gaussien
                                       multivarié de covariance isotrope.
    ================================== ========================================================================
'''

# Version 5

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

import maybe_useful as mbu

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions d'estimations
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def L_opt(S,L_min=1,detail=False):
    '''Nombre optimal de dimensions latentes, estimé par la "méthode du saut".
    
    Paramètres
    ----------
    S : 2-D ndarray,
        Matrice de covariance empirique (symétrique positive d'ordre D) à diagonaliser.
        
    L_min : int, optional,
        Nombre de dimensions minimal à renvoyer.
        Mis sur 1 par défaut.
        
    detail : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, détaille graphiquement le choix du nombre de dimensions latentes.
        
        Le coefficient d'incertitude donné est égal à log(total_span/taken_span)/log(D) où
            - total_span est la différence entre la plus grande et la plus petite valeur propre de S.
            - taken_span est la différence entre la L-ième et la (L+1)-ème plus grande valeur propre de S.
        
    Renvois
    -------
    L : int,
        Nombre de dimensions latentes pour la matrice de covariance S minimisant le coefficient d'incertitude.
    '''
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
    '''Nombre optimal de dimensions latentes, estimé par la "méthode du coude".
    
    Paramètres
    ----------
    S : 2-D ndarray,
        Matrice de covariance empirique (symétrique positive d'ordre D) à diagonaliser.
        
    beta : float, optional,
        Coefficient pour ajuster la part d'importance des abscisses et des ordonnées dans le calcul de l'emplacement du "coude".
        Plus beta est grand, plus la part d'importance des ordonnées dans le calcul de l'emplacement du "coude" sera grande, et donc plus la valeur de L renvoyée sera élevée.
        Mis sur 0.5 par défaut.
        
    L_min : int, optional,
        Nombre de dimensions minimal à renvoyer.
        Mis sur 1 par défaut.
        
    detail : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, détaille graphiquement le choix du nombre de dimensions latentes.
        
        Le coefficient d'incertitude donné est égal à log(total_span/taken_span)/log(D) où
            - total_span est la différence entre la plus grande et la plus petite valeur propre de S.
            - taken_span est la différence entre la L-ième et la (L+1)-ème plus grande valeur propre de S.
        
    Renvois
    -------
    L : int,
        Nombre de dimensions latentes correspondant à l'emplacement du coude dans le spectre ordonné de S.
    '''
    D1,D2 = np.shape(S)
    if D1 != D2:
        print('Erreur de dimensions sur S')
    else :
        D = D1
        SpS,P = nla.eig(S)
        
        ordre = np.sort(SpS)
        if L_min <= 1:
            L_star = D - mbu.R_elbow(ordre,beta) - 1
        else :
            L_star = D - mbu.R_elbow(ordre[:1-L_min],beta) - 1
        
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
    '''Nombre optimal de dimensions latentes, estimé par la "méthode du genou".
    
    Paramètres
    ----------
    S : 2-D ndarray,
        Matrice de covariance empirique (symétrique positive d'ordre D) à diagonaliser.
        
    beta : float, optional,
        Coefficient pour ajuster la part d'importance des abscisses et des ordonnées dans le calcul de l'emplacement du "genou".
        Plus beta est grand, plus la part d'importance des ordonnées dans le calcul de l'emplacement du "genou" sera grande, et donc plus la valeur de L renvoyée sera élevée.
        Mis sur 0.5 par défaut.
        
    L_min : int, optional,
        Nombre de dimensions minimal à renvoyer.
        Mis sur 1 par défaut.
        
    detail : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, détaille graphiquement le choix du nombre de dimensions latentes.
        
        Le coefficient d'incertitude donné est égal à log(total_span/taken_span)/log(D) où
            - total_span est la différence entre la plus grande et la plus petite valeur propre de S.
            - taken_span est la différence entre la L-ième et la (L+1)-ème plus grande valeur propre de S.
        
    Renvois
    -------
    L : int,
        Nombre de dimensions latentes correspondant à l'emplacement du genou dans le scree plot induit par S.
    '''
    D1,D2 = np.shape(S)
    if D1 != D2:
        print('Erreur de dimensions sur S')
    else :
        D = D1
        SpS,P = nla.eig(S)
        
        ordre = np.sort(SpS)
        inertie = np.cumsum(ordre)
        
        if L_min <= 1:
            L_star = D - mbu.R_elbow(inertie,beta) - 1
        else :
            L_star = D - mbu.R_elbow(inertie[:1-L_min],beta) - 1
        
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

def PCA(Y,L=None):
    '''Analyse en Composante Principale.
    (Adapté pour le modèle (M.1))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés.
    
    L : int, optional,
        Nombre de dimensions latentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de L renvoyée par la fonction L_opt
        
    Renvois
    -------
    S : 2-D ndarray,
        Matrice de covariance des vecteurs constituant les lignes de Y.
        
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux obtenus par la PCA.
        
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la PCA.
    '''
    N,D = np.shape(Y)
    
    #Centrage de Y
    mu_ML = np.mean(Y,axis=0)
    Yc = np.array([y-mu_ML for y in Y])
    
    #Diagonalisation de S, choix des axes principaux
    S = mbu.cov_emp(Yc)
    
    if type(L) == type(None):
        L = L_opt(S)
    
    SpS, P = nla.eig(S)
    ordre = np.sort(SpS)
    inlist = [k for k in range(D) if SpS[k] in ordre[D-L:]]
    
    #Estimation de W
    P = mbu.normalize(P)
    tW = np.array([P[:,k] for k in inlist])
    W = np.transpose(tW)
    
    #Estimation de Z
    Z = Yc @ W @ nla.inv(tW@W)
    
    return S, W, Z

def bruit(S,L=None):
    '''Estimation de la variance du bruit à partir d'une matrice de covariance.
    
    Paramètres
    ----------
    S : 2-D ndarray,
        Matrice de covariance empirique (symétrique positive d'ordre D) à diagonaliser.
    
    L : int, optional
        Nombre de dimensions latentes supposé.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de L renvoyée par la fonction L_opt
        
    Renvois
    -------
    sigma2 : float,
        Bruit estimé à partir de la matrice de covariance S, égal à la moyenne des D-L plus petites valeurs propres de S.
    '''
    D1,D2 = np.shape(S)
    
    if type(L) == type(None):
        L = L_opt(S)
        
    if D1 != D2 or D1 <= L or D2 <= L :
        print('Erreur de dimension sur S')
        
    else :
        D = D1
        SpS, P = nla.eig(S)
        sigma2 = np.mean(np.sort(SpS)[D-L:])
        return sigma2

def PPCA_EM(Y,L=None,nb_steps=1000,err=0.0,tempo=True):
    '''Analyse en Composante Principale Probabiliste, algorithme E-M.
    (Adapté pour le modèle (M.1))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés.
    
    L : int, optional,
        Nombre de dimensions latentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de L renvoyée par la fonction L_opt
        
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme.
        Mis sur 1000 par défaut.
        
    err : float, optional,
        Erreur en distance L2 entre deux solutions consécutives en-dessous de laquelle l'algorithme s'arrête.
        Mis sur 0.0 par défaut.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme.
        Mis sur True par défaut.
        
    Renvois
    -------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux obtenus par la PPCA.
        
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la PPCA.
        
    sigma2 : float,
        Estimation de la variance du bruit.
    '''
    N,D = np.shape(Y)
    
    #Centrage de Y
    mu_Y = np.mean(Y,axis=0)
    Yc = np.array([y-mu_Y for y in Y])
    
    #Initialisation
    S = mbu.cov_emp(Y)
    
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

def PPCA(Y,L=None):
    '''Analyse en Composante Principale, algorithme direct.
    (Adapté pour le modèle (M.1))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés.
    
    L : int, optional,
        Nombre de dimensions latentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de L renvoyée par la fonction L_opt.
        
    Renvois
    -------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux obtenus par la PCA.
        
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la PCA.
    
    sigma2 : float,
        Estimation de la variance du bruit.
    '''
    N,D = np.shape(Y)
    
    #Centrage de Y
    mu_ML = np.mean(Y,axis=0)
    Yc = np.array([y-mu_ML for y in Y])
    
    #Estimation de sigma²
    S = mbu.cov_emp(Y)    
    if type(L) == type(None):
        L = L_opt(S)
    
    SpS, P = nla.eig(S)
    P = mbu.normalize(P)
    ordre = np.sort(SpS)
    sigma2 = np.mean(ordre[:D-L])
    inlist = [k for k in range(D) if SpS[k] in ordre[D-L:]]
    
    #Estimation de W et Z
    tU_L = np.array([P[:,k] for k in inlist])
    L_L = np.diag(np.array([np.sqrt(SpS[k]-sigma2) for k in inlist]))
    tW = L_L @ tU_L
    W = np.transpose(tW)
    Z = Yc @ W @ nla.inv(tW@W)
    
    return W, Z, sigma2

def ML_RCA(Y,X,L,V=None,nb_steps=100,err=0.0,tempo=True):
    '''Analyse en Composante Résiduelle, méthode itérative.
    (Adapté pour le modèle (M.2))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont des vecteurs de covariables.
        L'algorithme s'assure que les vecteurs de covariables sont centrés en les centrant de force.
    
    L : int,
        Nombre de dimensions latentes souhaité.
        
    V : 2-D ndarray, optional,
        Matrice de taille (D,C) d'effets fixes donnée en argument initial de l'algorithme itératif.
    
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme.
        Mis sur 1000 par défaut.
        
    err : float, optional,
        Erreur en distance L2 entre deux solutions consécutives en-dessous de laquelle l'algorithme s'arrête.
        Mis sur 0.0 par défaut.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme.
        Mis sur True par défaut.
        
    Renvois
    -------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux obtenus par la RCA.
        
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la RCA.
        
    V_hat : 2-D ndarray,
        Matrice de taille (D,C) d'effets fixes estimée par la RCA.
    
    sigma2 : float,
        Estimation de la variance du bruit.
    '''
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
        
        #Estimation de Z et W
        W,Z,sigma2 = PPCA(Yc-Xc@np.transpose(V_hat),L)
        
    #Si V n'est pas donné
    else :
        #Initialisation
        
        #Estimation de V
        V_hat = np.transpose(Yc)@Xc @ nla.inv(np.transpose(Xc)@Xc)
        
        #Estimation de Z et W
        W,Z,sigma2 = PPCA(Yc-Xc@np.transpose(V_hat),L)
        
        dist = err+1
        t = 0
        
        #Boucle
        while dist > err and t < nb_steps :
            
            #Estimation de V
            new_V = np.transpose(Yc-Z@np.transpose(W))@Xc @ nla.inv(np.transpose(Xc)@Xc)
            
            #Estimation de Z et W
            new_W,new_Z,sigma2_hat = PPCA(Yc - Xc@np.transpose(new_V),L)
            
            #Vérification
            dist = np.sum((Z@np.transpose(W) - new_Z@np.transpose(new_W))**2) + np.sum((V_hat-new_V)**2)
            t += 1
            
            W = new_W
            Z = new_Z
            V_hat = new_V
            
        if tempo :
            print('t =',t)
    
    return W, Z, V_hat, sigma2

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
        S_hat = mbu.cov_emp(Y)
        A = nla.inv(Sigma) @ S_hat
        
        #GEP
        SpA, P = nla.eig(A)
        ord_A = np.sort(SpA)
        inlist_A = [k for k in range(D) if SpA[k] in ord_A[D-L:]]
        P_W = mbu.normalize(np.transpose(np.array([P[:,k] for k in inlist_A])))
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
        covX = mbu.cov_emp(Xc)
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
        la_hat = D**2 * 2/mbu.trace(Lambda_hat)
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
                la_hat = D**2 * 2/mbu.trace(Lambda_hat)
        
        t += 1
        log_p = new_log_p
        
    if tempo :
        print('t =',t)
    
    #Estimation de Z
    Z = (Yc - Xc) @ W_hat @ nla.inv(np.transpose(W_hat)@W_hat)
    
    return W_hat,Z,Lambda_hat,sigma2

def MLE_Gauss(Y):
    '''Estimation de la moyenne et de la variance d'un N-échantillon Gaussien multivarié de covariance isotrope.
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés, i.i.d. de loi normale.
        
    Renvois
    -------
    mu : 1-D ndarray,
        Estimation de la moyenne de la loi des vecteurs lignes de Y.
    
    sigma2 : float,
        Estimation de la variance de la loi des vecteurs lignes de Y.
    '''
    N,D = np.shape(Y)
    mu_ML = np.mean(Y,axis=0)
    sigma2_ML = np.var(Y)*(N/(N-1))
    return mu_ML,sigma2_ML