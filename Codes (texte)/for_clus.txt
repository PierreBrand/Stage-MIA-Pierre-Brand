'''===================================
    Fonctions utiles pour le clustering
    ===================================
    
    ================================== =========================================================================
    Contient les fonctions suivantes :
    ---------------------------------- -------------------------------------------------------------------------
    d_SL                               Distance en Single-Linkage entre 2 clusters.
    d_CL                               Distance en Complete-Linkage entre 2 clusters.
    d_AL                               Distance en Average-Linkage entre 2 clusters.
    d_L2_Ward                          Distance de Ward entre 2 clusters.
    dissim_L2                          Matrice de dissimilarité en distance L2 entre les lignes d'une matrice.
    condense                           Condensation en vecteur d'une matrice symétrique.
    tri                                Tri des lignes d'une matrice selon un clustering donné.
    omegate                            Transformation d'une liste de clusters en un vecteur d'entiers.
    matrixage                          Transformation d'un vecteur contenant des numéros de clusters
                                       en une matrice contenant des 0 et des 1.
    occurences                         Nombre d'occurences de chaque coefficient dans un vecteur d'entiers.
    perm_opt                           Permutation optimale pour faire correspondre ensemble
                                       deux vecteurs d'entiers de même taille.
    Dist_CP                            Distance L2 minimale d'un vecteur à une liste de vecteurs de même taille.
    sil_coeff                          Coefficient silhouette d'un vecteur d'une matrice d'observations
                                       pour un clustering donné.
    sil_score                          Score silhouette d'une matrice observations pour un clustering donné.
    distorsion                         Distorsion d'une matrice d'observations pour un clustering donné.
    Lap                                Laplacienne normalisée d'une matrice de similarité.
    ARS                                Adjusted Rand Score entre deux vecteurs d'entiers de même taille.
    ================================== =========================================================================
'''

#Version 3

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
#Fonctions utiles pour le clustering
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Single-linkage
def d_SL(Pdm,G,H):
    '''Distance en Single-Linkage entre les clusters G et H.
    
    Paramètres
    ----------
    Pdm : 2-D ndarray,
        Matrice symétrique d'ordre N de dissimilarité dont chaque coefficient est un réel positif quantifiant la dissimilarité entre deux individus (une distance par exemple). 
    
    G : list,
        Liste des numéros des individus du cluster G, devant être compris entre 0 et N-1.
        
    H : list,
        Liste des numéros des individus du cluster H, devant être compris entre 0 et N-1.
    
    Renvois
    -------
    dist : float,
        Distance en Single-Linkage entre les clusters G et H.
    '''
    if np.any(Pdm-np.transpose(Pdm)) or np.any(Pdm<0):
        print("Pdm n'est pas une matrice de dissimilarité")
    else :
        N,N1 = np.shape(Pdm)
        dist_tab = np.array([[Pdm[i][j] for j in H] for i in G])
        return np.min(dist_tab)

#Complete-linkage
def d_CL(Pdm,G,H):
    '''Distance en Complete-Linkage entre les clusters G et H.
    
    Paramètres
    ----------
    Pdm : 2-D ndarray,
        Matrice symétrique d'ordre N de dissimilarité dont chaque coefficient est un réel positif quantifiant la dissimilarité entre deux individus (une distance par exemple). 
    
    G : list,
        Liste des numéros des individus du cluster G, devant être compris entre 0 et N-1.
        
    H : list,
        Liste des numéros des individus du cluster H, devant être compris entre 0 et N-1.
    
    Renvois
    -------
    dist : float,
        Distance en Complete-Linkage entre les clusters G et H.
    '''
    if np.any(Pdm-np.transpose(Pdm)) or np.any(Pdm<0):
        print("Pdm n'est pas une matrice de dissimilarité")
    else :
        N,N1 = np.shape(Pdm)
        dist_tab = np.array([[Pdm[i][j] for j in H] for i in G])
        return np.max(dist_tab)

#Average-linkage
def d_AL(Pdm,G,H):
    '''Distance en Average-Linkage entre les clusters G et H.
    
    Paramètres
    ----------
    Pdm : 2-D ndarray,
        Matrice symétrique d'ordre N de dissimilarité dont chaque coefficient est un réel positif quantifiant la dissimilarité entre deux individus (une distance par exemple). 
    
    G : list,
        Liste des numéros des individus du cluster G, devant être compris entre 0 et N-1.
        
    H : list,
        Liste des numéros des individus du cluster H, devant être compris entre 0 et N-1.
    
    Renvois
    -------
    dist : float,
        Distance en Average-Linkage entre les clusters G et H.
    '''
    if np.any(Pdm-np.transpose(Pdm)) or np.any(Pdm<0):
        print("Pdm n'est pas une matrice de dissimilarité")
    else :
        N,N1 = np.shape(Pdm)
        dist_tab = np.array([[Pdm[i][j] for j in H] for i in G])
        return np.mean(dist_tab)

#Distance Ward
def d_L2_Ward(X,G,H):
    '''Distance de Ward entre les clusters G et H.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les individus à clusteriser.
    
    G : list,
        Liste des numéros des individus du cluster G, devant être compris entre 0 et N-1.
        
    H : list,
        Liste des numéros des individus du cluster H, devant être compris entre 0 et N-1.
    
    Renvois
    -------
    dist : float,
        Distance de Ward entre les clusters G et H.
    '''    
    N,D = np.shape(X)
    
    if max(G) >= N or max(H) >= N :
        print("Pas assez d'individus")
    else :
        mu_G = np.mean(np.array([X[n] for n in G]),axis=0)
        mu_H= np.mean(np.array([X[n] for n in H]),axis=0)
        return np.sum((mu_G - mu_H)**2)

#Distance L2
def dissim_L2(X):
    '''Matrice de dissimilarité en distance L2 des lignes de X.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des individus.
    
    Renvois
    -------
    Pdm : 2-D ndarray,
        Matrice carré d'ordre N, dont, pour tous i,j entre 0 et N-1, le coefficient à la i-ème ligne et la j-ième colonne est la distance L2 entre le i-ème et le j-ème vecteur ligne de X.
    '''
    N,D = np.shape(X)
    return np.array([[np.sqrt(np.sum((X[i]-X[j])**2)) for i in range(N)] for j in range(N)])

#Condensation d'une matrice de dissimilarité
def condense(PdM):
    '''Condensation d'une matrice symétrique.
    
    Paramètres
    ----------
    PdM : 2-D ndarray,
        Matrice symétrique d'ordre N.
    
    Renvois
    -------
    conc : 1-D ndarray,
        Concaténation des lignes de la partie triangulaire strictement supérieure de PdM.
    '''
    N,N1 = np.shape(PdM)
    return np.concatenate([PdM[n][n+1:] for n in range(N-1)])

#Tri des vecteurs
def tri(X,omega,K=None):
    '''Tri des vecteurs de X selon omega.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les individus à trier.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    K : int, optional,
        Mis à défaut sur la valeur maximale de omega + 1
        Si renseigné et plus grand que la valeur maximale de omega + 1, rendra des clusters vides.
        Si renseigné et plus petit que la valeur maximale de omega + 1, ne prendra pas en compte les individus dont le numéro de cluster est plus grand que K-1
    
    Renvois
    -------
    tri_X : list of ndarray,
        Liste à K éléments dont chaque élément est la matrice de taille (N_k,D) dont les lignes sont les individus d'un même cluster.
    '''
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
    '''Transformation d'une liste de clusters en un vecteur d'entiers.
    
    Paramètres
    ----------
    
    clusters : list of list,
        Liste de K listes, où, pour tout k entre 0 et K-1, le k-ième liste contient les numéros des individus appartenant au k-ième cluster.
    
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    K = len(clusters)
    N = max([max(clus) for clus in clusters]) + 1
    omega = np.zeros(N)
    
    for k in range(K) :
        for n in clusters[k] :
            omega[n] = k
    
    return omega.astype(int)

def matrixage(omega):
    '''Transformation d'un vecteur contenant des numéros de clusters en une matrice avec des 0 et des 1.
    
    Paramètres
    ----------
    omega : 1-D ndarray,
        Vecteur d'entiers de taille N et de valeur maximale K-1.
        
    Renvois
    -------
    O : 2-D ndarray,
        Matrice de taille (N,K), dont, pour tout n entre 0 et N-1 et tout k entre 0 et K-1, le coefficient à la n-ième ligne et la k-ième colonne vaut 1 si le n-ième coefficient de omega vaut k et 0 sinon.
    '''
    N = len(omega)
    K = int(np.max(omega) + 1)
    O = np.array([[int(omega[n] == k) for k in range(K)] for n in range(N)])
    
    return O

#Occurences
def occurences(omega):
    '''Nombres d'occurences de chaque coefficient d'un vecteur d'entiers.
    
    Paramètres
    ----------
    omega : 1-D ndarray,
        Vecteur d'entiers de taille N et de valeur maximale K-1.
        
    Renvois
    -------
    occ : 1-D ndarray,
        Vecteur de taille K, dont, pour tout k entre K-1, le k-ième coefficient de omega est le nombre d'occurences de k dans omega.
    '''
    N = len(omega)
    K = int(np.max(omega)) + 1
    
    occur = np.zeros(K)
    for n in range(N):
        occur[omega[n]] += 1
    
    return occur.astype(int)

def perm_opt(omega1,omega2):
    '''Permutation optimale pour faire correspondre ensemble deux vecteurs d'entiers de même taille.
    
    Paramètres
    ----------
    omega1 : 1-D ndarray,
        Vecteur d'entiers de taille N et de valeur maximale K-1.
        
    omega2 : 1-D ndarray,
        Vecteur d'entiers de taille N et de valeur maximale K-1.
        
    Renvois
    -------
    s : 1-D ndarray,
        Permutation pour laquelle le nombre de coefficients différant entre s(omega1) et omega2 est minimal.
    '''
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
    '''Distance L2 minimale d'un vecteur x à une liste M de vecteurs de même taille que x.
    
    Paramètres
    ----------
    x : 1-D ndarray,
        Vecteur de taille D.
        
    M : 2-D ndarray,
        Matrice de taille (t,D), dont les lignes sont les vecteurs dont il faut calculer la distance à x
    
    Renvois
    -------
    dist : float,
        Distance L2 de x au vecteur ligne de M qui lui est le plus proche.
    '''
    t,D = np.shape(M)
    D1 = len(x)
    if D != D1:
        print("x et les moyennes n'ont pas même dimensions")
    else :
        return min([np.sum((x-M[k])**2) for k in range(t)])

#Silhouette coefficient
def sil_coeff(n,X,omega):
    '''Coefficient silhouette du n-ième individu des observations X pour le clustering induit par omega.
    
    Paramètres
    ----------
    n : int,
        Numéro de l'individu dont on veut le coefficient silhouette. Doit être compris entre 0 et N-1.
        
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Renvois
    -------
    coeff : float,
        Coefficient silhouette du n-ième individu des observations X pour le clustering donné par omega.
    '''
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
    '''Score silhouette des observations X pour le clustering induit par omega.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Renvois
    -------
    score : float,
        Score silhouette des observations X pour le clustering donné par omega.
    '''
    N = len(X)
    return np.mean([sil_coeff(n,X,omega) for n in range(N)])

#Distorsion
def distorsion(X,omega):
    '''Distorsion des observations X pour le clustering induit par omega.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Renvois
    -------
    dist : float,
        Somme des erreurs en distance L2 de chaque individu au centre de son cluster.
    '''
    tri_X = tri(X,omega)
    return np.sum(np.array([np.sum(np.var(x,axis=0)) for x in tri_X]))

def Lap(Psm):
    '''Laplacienne normalisée d'une matrice de similarité
    
    Paramètres
    ----------
    Psm : 2-D ndarray,
        Matrice de similarité pairwise symétrique d'ordre N, dont les coefficients sont positifs et sont d'autant plus élevés que l'individu en abscisse et celui en ordonnée sont proches.
    
    Renvois
    -------
    Lapsym : 2-D ndarray,
        Matrice Laplacienne normalisée de Psm
    '''
    N,N1 = np.shape(Psm)
    
    if np.any(Psm-np.transpose(Psm)) or np.any(Psm<0):
        print("Psm n'est pas une matrice de poids symétrique")
    else :
        vec_D = np.sum(Psm, axis=0)
        rinv_D = np.diag(1/np.sqrt(vec_D))
        
        L = np.eye(N) - rinv_D @ Psm @ rinv_D
        
        return L

def ARS(omega1,omega2):
    '''Adjusted Rand Score entre deux vecteurs d'entiers de même taille.
    
    Paramètres
    ----------
    omega1 : 1-D ndarray,
        Vecteur d'entiers de taille N.
        
    omega2 : 1-D ndarray,
        Vecteur d'entiers de taille N.
        
    Renvois
    -------
    ars : float,
        Adjusted Rand Score entre les clusterings induits par omega1 et omega2.
    '''
    return sklmc.adjusted_rand_score(omega1,omega2)