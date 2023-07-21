'''===============================================
    Fonctions d'estimation pour mixtures de modèles
    ===============================================
    
    ================================== ===================================================================================
    Contient les fonctions suivantes :
    ---------------------------------- -----------------------------------------------------------------------------------
    obs1                               Clustering, puis PPCA appliquée sur chaque cluster (Adapté pour le modèle (M.4.1)).
    obs2                               Clustering, puis RCA appliquée sur chaque cluster (Adapté pour le modèle (M.4.2)).
    lat1                               PPCA, puis clustering (Adapté pour le modèle (M.5.1)).
    lat2                               RCA, puis clustering (Adapté pour le modèle (M.5.2)).
    ================================== ===================================================================================
'''

# Version 7

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions de MFA
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def obs1(Y,L,K=None,omega=None,fun=clf.Lapras,nb_steps=100,tempo=True):
    '''Clustering, puis PPCA appliquée sur chaque cluster.
    (Adapté pour le modèle (M.4.1))
    
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
    '''
    N,D = np.shape(Y)
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = clf.K_opt(Y)
                
        omega_hat = fun(Y,K,tempo=tempo)
        omega_hat = clf.K_means(Y,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    reclus = [[n for n in range(N) if omega_hat[n] == k] for k in range(K)]
    tri_Y = ufc.tri(Y,omega_hat)
    thetas_hat = [[] for k in range(K)]
    Z_hat = np.zeros((N,L))
    
    for k in range(K):
        
        y = tri_Y[k]
        card_k = len(y)
        mu_k = np.mean(y,axis=0)
        W_k_hat, Z_k_hat, sigma2_k_hat = sfa.PPCA(y,L)
        
        thetas_hat[k] = [W_k_hat, mu_k, sigma2_k_hat]
        for j in range(card_k):
            Z_hat[reclus[k][j]] = Z_k_hat[j]
        
    return thetas_hat, Z_hat, omega_hat

def obs2(Y,X,L,K=None,omega=None,fun=clf.Lapras,nb_steps=100,err=0.0,tempo=True):
    '''Clustering, puis RCA appliquée sur chaque cluster.
    (Adapté pour le modèle (M.4.2))
    
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
    '''
    N,D = np.shape(Y)
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = clf.K_opt(Y)
                
        omega_hat = fun(Y,K,tempo=tempo)
        omega_hat = clf.K_means(Y,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    reclus = [[n for n in range(N) if omega_hat[n] == k] for k in range(K)]
    tri_Y = ufc.tri(Y,omega_hat)
    tri_X = ufc.tri(X,omega_hat)
    thetas_hat = [[] for k in range(K)]
    Z_hat = np.zeros((N,L))
    
    for k in range(K):
        
        y = tri_Y[k]
        x = tri_X[k]
        card_k = len(y)
        mu_k = np.mean(y,axis=0)
        W_k_hat, Z_k_hat, V_k_hat, sigma2_k_hat = sfa.ML_RCA(y,x,L,nb_steps=nb_steps,err=err,tempo=tempo)
        
        thetas_hat[k] = [W_k_hat, V_k_hat, mu_k, sigma2_k_hat]
        for j in range(card_k):
            Z_hat[reclus[k][j]] = Z_k_hat[j]
        
    return thetas_hat, Z_hat, omega_hat

def lat1(Y,L=None,K=None,omega=None,fun=clf.Lapras,nb_steps=100,tempo=True,latent=True):
    '''PPCA, puis clustering.
    (Adapté pour le modèle (M.5.1))
    
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
    
    latent : bool, optional,
        Si mis sur False, le clustering se fait sur la vecteurs observés.
        Si mis sur True, le clustering se fait sur la vecteurs latents.
        Mis sur True par défaut.
    
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
    '''
    N,D = np.shape(Y)
    W_prov, Z_prov, sigma2_hat = sfa.PPCA(Y,L)
    D,L = np.shape(W_prov)
    
    D_W = np.diag(np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
    D_W_inv = np.diag(1/np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
    Z_hat = Z_prov @ D_W
    W_hat = W_prov @ D_W_inv
    
    mu_hat = np.mean(Y,axis=0)
    eta_hat = W_hat,mu_hat,sigma2_hat
    
    if type(omega) == type(None):
        
        if latent:
            if type(K) == type(None):
                K = clf.K_opt(Z_hat)

            omega_hat = fun(Z_hat,K,tempo=tempo)
            omega_hat = clf.K_means(Z_hat,omega_hat,nb_steps,tempo=tempo)
        else :
            if type(K) == type(None):
                K = clf.K_opt(Y)

            omega_hat = fun(Y,K,tempo=tempo)
            omega_hat = clf.K_means(Y,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    tri_Z = ufc.tri(Z_hat,omega_hat)
    thetas_hat = []
    
    for k in range(K):
        z = tri_Z[k]
        thetas_k = sfa.MLE_Gauss(z)
        thetas_hat.append(thetas_k)
        
    return eta_hat, thetas_hat, Z_hat, omega_hat

def lat2(Y,X,L,K=None,omega=None,fun=clf.Lapras,nb_steps=100,err=0.0,tempo=True):
    '''RCA, puis clustering.
    (Adapté pour le modèle (M.5.2))
    
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
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
    
    err : float, optional,
        Erreur en distance L2 entre deux solutions consécutives en-dessous de laquelle l'algorithme s'arrête.
        Mis sur 0.0 par défaut.
    
    latent : bool, optional,
        Si mis sur False, le clustering se fait sur la vecteurs observés.
        Si mis sur True, le clustering se fait sur la vecteurs latents.
        Mis sur True par défaut.
    
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
    '''
    N,D = np.shape(Y)
    N1,C = np.shape(X)
    W_prov, Z_prov, V_hat, sigma2_hat = sfa.ML_RCA(Y,X,L,nb_steps=nb_steps,err=err,tempo=tempo)
    
    D_W = np.diag(np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
    D_W_inv = np.diag(1/np.array([mbu.norme(W_prov[:,l]) for l in range(L)]))
    Z_hat = Z_prov @ D_W
    W_hat = W_prov @ D_W_inv
    
    mu_hat = np.mean(Y,axis=0)
    eta_hat = W_hat,V_hat,mu_hat,sigma2_hat
    
    if type(omega) == type(None):
        
        if type(K) == type(None):
            K = clf.K_opt(Z_hat)
                
        omega_hat = fun(Z_hat,K,tempo=tempo)
        omega_hat = clf.K_means(Z_hat,omega_hat,nb_steps,tempo=tempo)
    
    else:
        omega_hat = omega.copy()
        K = int(max(omega) + 1)
    
    tri_Z = ufc.tri(Z_hat,omega_hat)
    thetas_hat = []
    
    for k in range(K):
        z = tri_Z[k]
        thetas_k = sfa.MLE_Gauss(z)
        thetas_hat.append(thetas_k)
        
    return eta_hat, thetas_hat, Z_hat, omega_hat