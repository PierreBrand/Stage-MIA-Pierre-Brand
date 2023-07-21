'''======================================
    Fonctions de la sélection de variables
    ======================================
    
    ================================== ==========================================================================
    Contient les fonctions suivantes :
    ---------------------------------- --------------------------------------------------------------------------
    PPCA                               Analyse en Composante Principale, puis Sélection de Variables
                                       (Adapté pour le modèle (M.6.1)).
    RCA                                Analyse en Composante Résiduelle (méthode itérative), puis Sélection de Variables
                                       (Adapté pour le modèle (M.6.2)).
    lat1                               PPCA, puis Sélection de Variables, puis clustering
                                       (Adapté pour le modèle (M.6.3)).
    lat2                               RCA, puis Sélection de Variables, puis clustering
                                       (Adapté pour le modèle (M.6.4)).
    obs1                               Clustering, puis PPCA et Sélection de Variables appliquées sur chaque cluster
                                       (Adapté pour le modèle (M.6.5)).
    obs2                               Clustering, puis RCA et Sélection de Variables appliquées sur chaque cluster
                                       (Adapté pour le modèle (M.6.6)).
    spe1                               PPCA, puis clustering, puis Sélection de Variables
                                       (Adapté pour le modèle (M.6.7)).
    spe2                               RCA, puis clustering, puis Sélection de Variables
                                       (Adapté pour le modèle (M.6.8)).
    ================================== ==========================================================================
'''

# Version 4

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
import for_clus as ufc
import single_fa as sfa
import clus_funs as clf
import data_rnr as drr
import for_fs as ufs
import mixed_fa as mfa

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions de FS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def PPCA(Y,L=None,U=None):
    '''Analyse en Composante Principale, puis Sélection de Variables.
    (Adapté pour le modèle (M.6.1))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés.
    
    L : int, optional,
        Nombre de dimensions latentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de L renvoyée par la fonction L_opt.
        
    U : int, optional,
        Nombre de dimensions pertinentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de U renvoyée par la fonction U_opt.
        
    Renvois
    -------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux obtenus par la PCA.
        
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la PCA.
    
    sigma2 : float,
        Estimation de la variance du bruit.
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    '''
    W_hat,Z_hat,sigma2_hat = sfa.PPCA(Y,L)
    iota_hat = ufs.iotate(Y,Z_hat,U)
    
    U,L = np.shape(W_hat)
    N,D = np.shape(Y)
    Y_tilde = drr.discard(Y,iota_hat)
    
    W_hac,Z_hac,sigma2_hac = sfa.PPCA(Y_tilde,L)
    
    return W_hac,Z_hac,sigma2_hac,iota_hat

def RCA(Y,X,L,V=None,nb_steps=100,err=0.0,tempo=True,U=None):
    '''Analyse en Composante Résiduelle (méthode itérative), puis Sélection de Variables.
    (Adapté pour le modèle (M.6.2))
    
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
    
    U : int, optional,
        Nombre de dimensions pertinentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de U renvoyée par la fonction U_opt.
    
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
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    '''
    W_hat, Z_hat, V_hat, sigma2_hat = sfa.ML_RCA(Y,X,L,V,nb_steps,err,tempo)
    mu_X = np.mean(X,axis=0)
    Xc = X - mu_X
    
    iota_hat = ufs.iotate(Y-Xc@np.transpose(V_hat),Z_hat,U)
    D,C = np.shape(V_hat)
    U,L = np.shape(W_hat)
    N = len(Y)
    Y_tilde = drr.discard(Y-Xc@np.transpose(V_hat),iota_hat)
    
    W_hac,Z_hac,sigma2_hac = sfa.PPCA(Y_tilde,L)
    
    return W_hac, Z_hac, V_hat, sigma2_hac, iota_hat

def lat1(Y,L=None,K=None,omega=None,fun=clf.Lapras,nb_steps=100,tempo=True,U=None):
    '''PPCA, puis Sélection de Variables, puis clustering.
    (Adapté pour le modèle (M.6.3))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs observés.
    
    L : int, optional,
        Nombre de dimensions latentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de L renvoyée par la fonction L_opt.
    
    K : int, optional,
        Nombre de clusters à identifier.
        Si omega et K sont mis sur None, le nombre de clusters à identifier est estimé à partir de la fonction K_opt.
        Si omega est renseigné mais pas K, K prend la valuer du coefficient maximal de omega + 1.
        Mis sur None par défaut.
    
    omega : 1-D ndarray, optional,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
        Si mis sur None, omega est estimé à l'aide d'un algorithme de clustering.
        Mis sur None par défaut.
        
    fun : function, optional,
        Fonction de clustering à utiliser.
        Mis sur clf.Lapras() par défaut.
    
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme K-means.
        Mis sur 100 par défaut.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
    
    U : int, optional,
        Nombre de dimensions pertinentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de U renvoyée par la fonction U_opt.
    
    Renvois
    -------
    eta : list,
        Liste de paramètres de la forme [W,mu,sigma2] où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux.
            - mu est un vecteur de taille D, supposé être la moyenne des observations.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations.
            
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [mu,sigma2] où :
            - mu est un vecteur de taille D, supposé être la moyenne des vecteurs du cluster.
            - sigma2 est un réel positif, supposé être la variance des vecteurs du cluster.
            
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la PPCA.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    '''
    N,D = np.shape(Y)
    W_prov,Z_prov,sigma2_hat = sfa.PPCA(Y,L)
    D,L = np.shape(W_prov)
    
    D_W = np.diag(np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
    D_W_inv = np.diag(1/np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
    Z_hat = Z_prov @ D_W
    W_hat = W_prov @ D_W_inv
    
    mu_hat = np.mean(Y,axis=0)
    
    iota_hat = ufs.iotate(Y,Z_hat,U)
    
    U = np.sum(iota_hat)
    Y_tilde = drr.discard(Y,iota_hat)
    
    W_hac,Z_hac,sigma2_hac = sfa.PPCA(Y_tilde,L)
        
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = clf.K_opt(Z_hac)
                
        omega_hat = fun(Z_hac,K,tempo=tempo)
        omega_hat = clf.K_means(Z_hac,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    tri_Z = ufc.tri(Z_hac,omega_hat)
    thetas_hac = []
    
    for k in range(K):
        z = tri_Z[k]
        thetas_k = sfa.MLE_Gauss(z)
        thetas_hac.append(thetas_k)
    
    Y_hat = drr.FS_rec2(W_hac,Z_hac,mu_hat,iota_hat)
    sigma2_hac = np.mean((Y-Y_hat)**2)
    eta_hac = W_hac,mu_hat,sigma2_hac
    
    return eta_hac, thetas_hac, Z_hac, omega_hat, iota_hat

def lat2(Y,X,L,K=None,omega=None,fun=clf.Lapras,nb_steps=100,err=0.0,tempo=True,U=None):
    '''RCA, puis Sélection de Variables, puis clustering.
    (Adapté pour le modèle (M.6.4))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs observés.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont des vecteurs de covariables.
        L'algorithme s'assure que les vecteurs de covariables sont centrés en les centrant de force.
    
    L : int,
        Nombre de dimensions latentes souhaité.
    
    K : int, optional,
        Nombre de clusters à identifier.
        Si omega et K sont mis sur None, le nombre de clusters à identifier est estimé à partir de la fonction K_opt.
        Si omega est renseigné mais pas K, K prend la valuer du coefficient maximal de omega + 1.
        Mis sur None par défaut.
    
    omega : 1-D ndarray, optional,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
        Si mis sur None, omega est estimé à l'aide d'un algorithme de clustering.
        Mis sur None par défaut.
        
    fun : function, optional,
        Fonction de clustering à utiliser.
        Mis sur clf.Lapras() par défaut.
    
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme K-means.
        Mis sur 100 par défaut.
    
    err : float, optional,
        Erreur en distance L2 entre deux solutions consécutives en-dessous de laquelle l'algorithme s'arrête.
        Mis sur 0.0 par défaut.
    
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
    
    U : int, optional,
        Nombre de dimensions pertinentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de U renvoyée par la fonction U_opt.
    
    Renvois
    -------
    eta : list,
        Liste de paramètres de la forme [W,mu,sigma2] où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux.
            - mu est un vecteur de taille D, supposé être la moyenne des observations.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations.
            
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [mu,sigma2] où :
            - mu est un vecteur de taille D, supposé être la moyenne des vecteurs du cluster.
            - sigma2 est un réel positif, supposé être la variance des vecteurs du cluster.
            
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la PPCA.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    '''
    N,D = np.shape(Y)
    N1,C = np.shape(X)
    W_prov, Z_prov, V_hat, sigma2_hat = sfa.ML_RCA(Y,X,L,nb_steps=nb_steps,err=err,tempo=tempo)
    
    D_W = np.diag(np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
    D_W_inv = np.diag(1/np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
    Z_hat = Z_prov @ D_W
    W_hat = W_prov @ D_W_inv
    
    mu_hat = np.mean(Y,axis=0)
    mu_X = np.mean(X,axis=0)
    Xc = X - mu_X
    
    iota_hat = ufs.iotate(Y-Xc@np.transpose(V_hat),Z_hat,U)
    
    U = np.sum(iota_hat)
    Y_tilde = drr.discard(Y-Xc@np.transpose(V_hat),iota_hat)
    
    W_hac,Z_hac,sigma2_hac = sfa.PPCA(Y_tilde,L)
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = clf.K_opt(Z_hac)
                
        omega_hat = fun(Z_hac,K,tempo=tempo)
        omega_hat = clf.K_means(Z_hac,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    tri_Z = ufc.tri(Z_hac,omega_hat)
    thetas_hat = []
    
    for k in range(K):
        z = tri_Z[k]
        thetas_k = sfa.MLE_Gauss(z)
        thetas_hat.append(thetas_k)
    
    Y_hat = drr.FS_rec2(W_hac,Z_hac,V_hat,X,mu_hat,iota_hat)
    sigma2_hac = np.mean((Y-Y_hat)**2)
    eta_hac = W_hac,V_hat,mu_hat,sigma2_hac
        
    return eta_hac, thetas_hat, Z_hac, omega_hat, iota_hat

def obs1(Y,L,K=None,omega=None,fun=clf.Lapras,nb_steps=100,tempo=True,U=None):
    '''Clustering, puis PPCA appliquée sur chaque cluster, puis Sélection de Variables sur chaque cluster.
    (Adapté pour le modèle (M.6.5))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs observés.
    
    L : int,
        Nombre de dimensions de l'espace latent.
    
    K : int, optional,
        Nombre de clusters à identifier.
        Si omega et K sont mis sur None, le nombre de clusters à identifier est estimé à partir de la fonction K_opt.
        Si omega est renseigné mais pas K, K prend la valuer du coefficient maximal de omega + 1.
        Mis sur None par défaut.
    
    omega : 1-D ndarray, optional,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
        Si mis sur None, omega est estimé à l'aide d'un algorithme de clustering.
        Mis sur None par défaut.
        
    fun : function, optional,
        Fonction de clustering à utiliser.
        Mis sur clf.Lapras() par défaut.
    
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme K-means.
        Mis sur 100 par défaut.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
    
    U : int, optional,
        Nombre de dimensions pertinentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de U renvoyée par la fonction U_opt.
    
    Renvois
    -------
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [W,mu,sigma2] où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux du cluster.
            - mu est un vecteur de taille D, supposé être la moyenne des observations du cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations du cluster.
            
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la PPCA, cluster par cluster.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    iotas : 2-D ndarray,
        Matrice de taille (K,D) dont chaque ligne contient U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    '''
    N,D = np.shape(Y)
    thetas_hat, Z_hat, omega_hat = mfa.obs1(Y,L,K,omega,fun,nb_steps,tempo)
    
    K = len(thetas_hat)
    tri_Y = ufc.tri(Y,omega_hat)
    tri_Z = ufc.tri(Z_hat,omega_hat)
    
    iotas_hat = np.zeros((K,D)).astype(int)
    reclus = ufc.tri((np.arange(N)).astype(int),omega_hat)
    Z_hac = np.zeros((N,L))
    thetas_hac = [[] for k in range(K)]
    
    for k in range(K):
        
        Y_k = tri_Y[k]
        Z_k = tri_Z[k]
        card_k = len(Y_k)
        
        mu_k_hac = np.mean(Y_k,axis=0)
        iota_k_hat = ufs.iotate(Y_k,Z_k,U)
        iotas_hat[k] = iota_k_hat
        U = np.sum(iota_k_hat)
        
        Y_k_tilde = drr.discard(Y_k,iota_k_hat)
        W_k_hac, Z_k_hac, sigma2_k_hac = sfa.PPCA(Y_k_tilde,L)
        
        for j in range(card_k):
            Z_hac[reclus[k][j]] = Z_k_hac[j]
        
        Y_k_hac = drr.FS_rec1(W_k_hac,Z_k_hac,mu_k_hac,iota_k_hat)
        sigma2_k_hac = np.mean((Y_k-Y_k_hac)**2)
            
        thetas_hac[k] = [W_k_hac,mu_k_hac,sigma2_k_hac]
        
    return thetas_hac, Z_hac, omega_hat, iotas_hat

def obs2(Y,X,L,K=None,omega=None,fun=clf.Lapras,nb_steps=100,err=0.0,tempo=True,U=None):
    '''Clustering, puis RCA appliquée sur chaque cluster, puis Sélection de Variables sur chaque cluster.
    (Adapté pour le modèle (M.6.6))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs observés.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont des vecteurs de covariables.
        L'algorithme s'assure que les vecteurs de covariables sont centrés en les centrant de force, cluster par cluster.
    
    L : int,
        Nombre de dimensions latentes souhaité.
    
    K : int, optional,
        Nombre de clusters à identifier.
        Si omega et K sont mis sur None, le nombre de clusters à identifier est estimé à partir de la fonction K_opt.
        Si omega est renseigné mais pas K, K prend la valuer du coefficient maximal de omega + 1.
        Mis sur None par défaut.
    
    omega : 1-D ndarray, optional,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
        Si mis sur None, omega est estimé à l'aide d'un algorithme de clustering.
        Mis sur None par défaut.
        
    fun : function, optional,
        Fonction de clustering à utiliser.
        Mis sur clf.Lapras() par défaut.
    
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme K-means.
        Mis sur 100 par défaut.
        
    err : float, optional,
        Erreur en distance L2 entre deux solutions consécutives en-dessous de laquelle l'algorithme s'arrête.
        Mis sur 0.0 par défaut.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
    
    U : int, optional,
        Nombre de dimensions pertinentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de U renvoyée par la fonction U_opt.
    
    Renvois
    -------
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [W,V,mu,sigma2] où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux du cluster.
            - V est la matrice de taille (D,C) d'effets fixes du cluster.
            - mu est un vecteur de taille D, supposé être la moyenne des observations du cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations du cluster.
            
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents obtenus par la RCA, cluster par cluster.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    iotas : 2-D ndarray,
        Matrice de taille (K,D) dont chaque ligne contient U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    '''
    N,D = np.shape(Y)
    thetas_hat, Z_hat, omega_hat = mfa.obs2(Y,X,L,K,omega,fun,nb_steps,tempo)
    
    K = len(thetas_hat)
    tri_Y = ufc.tri(Y,omega_hat)
    tri_Z = ufc.tri(Z_hat,omega_hat)
    tri_X = ufc.tri(X,omega_hat)
    
    iotas_hat = np.zeros((K,D)).astype(int)
    reclus = ufc.tri((np.arange(N)).astype(int),omega_hat)
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
        
        iota_k_hat = ufs.iotate(Y_k-Xc_k@np.transpose(V_k_hat),Z_k,U)
        iotas_hat[k] = iota_k_hat
        U = np.sum(iota_k_hat)
        
        Y_k_tilde = drr.discard(Y_k-Xc_k@np.transpose(V_k_hat),iota_k_hat)
        W_k_hac, Z_k_hac, sigma2_k_hac = sfa.PPCA(Y_k_tilde,L)
        
        for j in range(card_k):
            Z_hac[reclus[k][j]] = Z_k_hac[j]
        
        Y_k_hac = drr.FS_rec2(W_k_hac,Z_k_hac,V_k_hat,Xc_k,mu_k_hac,iota_k_hat)
        sigma2_k_hac = np.mean((Y_k-Y_k_hac)**2)
            
        thetas_hac[k] = [W_k_hac,V_k_hat,mu_k_hac,sigma2_k_hac]
        
    return thetas_hac, Z_hac, omega_hat, iotas_hat

def spe1(Y,L=None,K=None,omega=None,fun=clf.Lapras,tempo=True,latent=True,Dv=None,Lv=None):
    '''PPCA, puis clustering, puis Sélection de Variables.
    (Adapté pour le modèle (M.6.7))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs observés.
    
    L : int, optional,
        Nombre de dimensions latentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de L renvoyée par la fonction L_opt.
    
    K : int, optional,
        Nombre de clusters à identifier.
        Si omega et K sont mis sur None, le nombre de clusters à identifier est estimé à partir de la fonction K_opt.
        Si omega est renseigné mais pas K, K prend la valuer du coefficient maximal de omega + 1.
        Mis sur None par défaut.
    
    omega : 1-D ndarray, optional,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
        Si mis sur None, omega est estimé à l'aide d'un algorithme de clustering.
        Mis sur None par défaut.
        
    fun : function, optional,
        Fonction de clustering à utiliser.
        Mis sur clf.Lapras() par défaut.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
    
    latent : bool, optional,
        Si mis sur False, le clustering se fait sur la vecteurs observés.
        Si mis sur True, le clustering se fait sur la vecteurs latents.
        Mis sur True par défaut.
    
    Dv : int, optional,
        Nombre de dimensions pertinentes de l'espace observé souhaité.
        Doit être inférieur ou égal à D.
        Mis sur None par défaut.
        Si mis sur None ou strictement supérieur à D, utilise la valeur de Dv renvoyée par la fonction U_opt.
    
    Lv : int, optional,
        Nombre de dimensions pertinentes de l'espace observé souhaité.
        Doit être inférieur ou égal à L.
        Mis sur None par défaut.
        Si mis sur None ou strictement supérieur à L, utilise la valeur de Lv renvoyée par la fonction U_opt.
    
    Renvois
    -------
    eta : list,
        Liste de paramètres de la forme [Wv,Wu,mu,nu,sigma2,tau2], où :
            - Wv est une matrice de taille (Dv,Lv) dont les colonnes sont les axes principaux des dimensions "mixtes".
            - Wu est une matrice de taille (Du,Lu) dont les colonnes sont les axes principaux des dimensions "non-mixtes".
            - mu est un vecteur de taille Dv+Du, supposé être la moyenne des vecteurs observés.
            - nu est un vecteur de taille Lu, supposé être la moyenne des vecteurs latents dont la loi ne change pas selon le cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des vecteurs observés.
            - tau2 est un réel positif, supposé être la variance des vecteurs latents dont la loi ne change pas selon le cluster.
        
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [mu,sigma2] où :
            - mu est un vecteur de taille D, supposé être la moyenne des vecteurs du cluster.
            - sigma2 est un réel positif, supposé être la variance des vecteurs du cluster.
            
    Zv : 2-D ndarray,
        Matrice de taille (N,Lv) dont les lignes sont les vecteurs latents obtenus par la PPCA.
    
    Zu : 2-D ndarray,
        Matrice de taille (N,Lu) dont les lignes sont les vecteurs latents obtenus par la PPCA.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la loi des variables aléatoires diffère selon les clusters sur la dimension correspondante, et le nombre 0 signifie qu'elle ne diffère pas selon les clusters sur la dimension correspondante.
    '''
    N,D = np.shape(Y)    
    mu_hat = np.mean(Y,axis=0)
    
    if type(L) != type(None) and L >= D:
        print("L >= D")
        L = None
    
    if type(Dv) != type(None) and Dv > D:
        print("Dv > D")
        Dv = None
    
    if latent :
        
        W_prov, Z_prov, sigma2_hat = sfa.PPCA(Y,L)
        N,L = np.shape(Z_prov)
        
        D_W = np.diag(np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
        D_W_inv = np.diag(1/np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
        Z_hat = Z_prov @ D_W
        W_hat = W_prov @ D_W_inv
        
        if type(Lv) != type(None) and Lv > L:
            print("Lv > L")
            Lv = None
            
        if type(omega) == type(None):
            if type(K) == type(None):
                K = clf.K_opt(Z_hat)
            omega_hat = fun(Z_hat,K,tempo=tempo)
        else:
            omega_hat = omega
        
        iota_lat = ufs.iotate(Z_hat,omega_hat,Lv)
        Lv = np.sum(iota_lat)
        Lu = L-Lv
        tZu_hat,tZv_hat = ufc.tri(np.transpose(Z_hat),iota_lat)
        Zu_hat = np.transpose(tZu_hat)
        Zv_hat = np.transpose(tZv_hat)
        
        iota_hat = ufs.iotate_2(Y,Zv_hat,Dv)
        
    else:
        if type(omega) == type(None):
            if type(K) == type(None):
                K = clf.K_opt(Y)
            omega_hat = fun(Y,K,tempo=tempo)
        else:
            omega_hat = omega
        iota_hat = ufs.iotate(Y,omega_hat,Dv)
    
    Dv = np.sum(iota_hat)
    tYu,tYv = ufc.tri(np.transpose(Y),iota_hat)
    Yu = np.transpose(tYu)
    Yv = np.transpose(tYv)

    etav_hat, thetas_hat, Zv_hat, omega_hat = mfa.lat1(Yv,L=Lv,omega=omega_hat,tempo=tempo)
    Wv_hat = etav_hat[0]
    
    if type(L) != None and type(Lv) != None and L>=Lv:
        Lu = L - Lv
    else :
        Lu = None
    
    Wu_hat, Zu_hat, tau2_usef = sfa.PPCA(Yu,Lu)
    nu_hat, tau2_hat = sfa.MLE_Gauss(Zu_hat)
    
    eta_hac = [Wv_hat,Wu_hat,mu_hat,nu_hat,0.0,tau2_hat]
    Y_hat = drr.FS_sperec1(eta_hac,Zv_hat,Zu_hat,iota_hat)
    eta_hac[4] = np.mean((Y-Y_hat)**2)
    
    return eta_hac, thetas_hat, Zv_hat, Zu_hat, omega_hat, iota_hat

def spe2(Y,X,L,K=None,omega=None,fun=clf.Lapras,V=None,nb_steps=100,err=0.0,tempo=True,latent=True,Dv=None,Lv=None):
    '''RCA, puis clustering, puis Sélection de Variables.
    (Adapté pour le modèle (M.6.8))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs observés.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont des vecteurs de covariables.
        L'algorithme s'assure que les vecteurs de covariables sont centrés en les centrant de force.
    
    L : int, optional,
        Nombre de dimensions latentes souhaité.
        Mis sur None par défaut.
        Si mis sur None, utilise la valeur de L renvoyée par la fonction L_opt.
    
    K : int, optional,
        Nombre de clusters à identifier.
        Si omega et K sont mis sur None, le nombre de clusters à identifier est estimé à partir de la fonction K_opt.
        Si omega est renseigné mais pas K, K prend la valuer du coefficient maximal de omega + 1.
        Mis sur None par défaut.
    
    omega : 1-D ndarray, optional,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
        Si mis sur None, omega est estimé à l'aide d'un algorithme de clustering.
        Mis sur None par défaut.
        
    fun : function, optional,
        Fonction de clustering à utiliser.
        Mis sur clf.Lapras() par défaut.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
    
    latent : bool, optional,
        Si mis sur False, le clustering se fait sur la vecteurs observés.
        Si mis sur True, le clustering se fait sur la vecteurs latents.
        Mis sur True par défaut.
    
    Dv : int, optional,
        Nombre de dimensions pertinentes de l'espace observé souhaité.
        Doit être inférieur ou égal à D.
        Mis sur None par défaut.
        Si mis sur None ou strictement supérieur à D, utilise la valeur de Dv renvoyée par la fonction U_opt.
    
    Lv : int, optional,
        Nombre de dimensions pertinentes de l'espace observé souhaité.
        Doit être inférieur ou égal à L.
        Mis sur None par défaut.
        Si mis sur None ou strictement supérieur à L, utilise la valeur de Lv renvoyée par la fonction U_opt.
    
    Renvois
    -------
    eta : list,
        Liste de paramètres de la forme [Wv,Wu,V,mu,nu,sigma2,tau2], où :
            - Wv est une matrice de taille (Dv,Lv) dont les colonnes sont les axes principaux des dimensions "mixtes".
            - Wu est une matrice de taille (Du,Lu) dont les colonnes sont les axes principaux des dimensions "non-mixtes".
            - V est une matrice de taille (D,C) d'effets fixes.
            - mu est un vecteur de taille Dv+Du, supposé être la moyenne des vecteurs observés.
            - nu est un vecteur de taille Lu, supposé être la moyenne des vecteurs latents dont la loi ne change pas selon le cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des vecteurs observés.
            - tau2 est un réel positif, supposé être la variance des vecteurs latents dont la loi ne change pas selon le cluster.
        
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [mu,sigma2] où :
            - mu est un vecteur de taille D, supposé être la moyenne des vecteurs du cluster.
            - sigma2 est un réel positif, supposé être la variance des vecteurs du cluster.
            
    Zv : 2-D ndarray,
        Matrice de taille (N,Lv) dont les lignes sont les vecteurs latents obtenus par la RCA.
    
    Zu : 2-D ndarray,
        Matrice de taille (N,Lu) dont les lignes sont les vecteurs latents obtenus par la RCA.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la loi des variables aléatoires diffère selon les clusters sur la dimension correspondante, et le nombre 0 signifie qu'elle ne diffère pas selon les clusters sur la dimension correspondante.
    '''
    N,D = np.shape(Y)
    N1,C = np.shape(X)
    mu_hat = np.mean(Y,axis=0)
    Xc = X - np.mean(X,axis=0)
    
    if type(Dv) != type(None) and Dv > D:
        print("Dv > D")
        Dv = None
    
    if type(Lv) != type(None) and Lv > L:
            print("Lv > L")
            Lv = None
    
    if latent :
        
        W_hat, Z_hat, V_hat, sigma2_hat = sfa.ML_RCA(Y,X,L,V,nb_steps,err,tempo)
            
        if type(omega) == type(None):
            if type(K) == type(None):
                K = clf.K_opt(Z_hat)
            omega_hat = fun(Z_hat,K,tempo=tempo)
        else:
            omega_hat = omega
        
        iota_lat = ufs.iotate(Z_hat,omega_hat,Lv)
        Lv = np.sum(iota_lat)
        Lu = L-Lv
        tZu_hat,tZv_hat = ufc.tri(np.transpose(Z_hat),iota_lat)
        Zu_hat = np.transpose(tZu_hat)
        Zv_hat = np.transpose(tZv_hat)
        iota_hat = ufs.iotate_2(Y-Xc@np.transpose(V_hat),Zv_hat,Dv)
        
    else:
        if type(omega) == type(None):
            if type(K) == type(None):
                K = clf.K_opt(Y)
            omega_hat = fun(Y,K,tempo=tempo)
        else:
            omega_hat = omega
        iota_hat = ufs.iotate(Y,omega_hat,Dv)
    
    Dv = np.sum(iota_hat)
    tYu,tYv = ufc.tri(np.transpose(Y-Xc@np.transpose(V_hat)),iota_hat)
    Yu = np.transpose(tYu)
    Yv = np.transpose(tYv)

    etav_hat, thetas_hat, Zv_hat, omega_hat = mfa.lat1(Yv,L=Lv,omega=omega_hat,tempo=tempo)
    Wv_hat = etav_hat[0]
    
    N,Lv = np.shape(Zv_hat)
    Lu = L-Lv
    
    Wu_hat, Zu_hat, tau2_usef = sfa.PPCA(Yu,Lu)
    nu_hat, tau2_hat = sfa.MLE_Gauss(Zu_hat)
    
    eta_hac = [Wv_hat,Wu_hat,V_hat,mu_hat,nu_hat,0.0,tau2_hat]
    Y_hat = drr.FS_sperec2(eta_hac,Zv_hat,Zu_hat,X,iota_hat)
    eta_hac[5] = np.mean((Y-Y_hat)**2)
    
    return eta_hac, thetas_hat, Zv_hat, Zu_hat, omega_hat, iota_hat