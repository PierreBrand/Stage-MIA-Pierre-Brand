'''======================================================
    Fonctions pour reconstruire ou représenter les données
    ======================================================
    
    ================================== ==================================================================================
    Contient les fonctions suivantes :
    ---------------------------------- ----------------------------------------------------------------------------------
    PCA_rec                            Reconstruction des vecteurs observés après (P)PCA
                                       (Adapté aux modèles (M.1) et (M.5.1)).
    RCA_rec                            Reconstruction des vecteurs observés après RCA
                                       (Adapté aux modèles (M.2) et (M.5.2)).
    MFA_rec_1                          Reconstruction des vecteurs observés après clustering
                                       puis (P)PCA sur les différents clusters (Adapté au modèle (M.4.1)).
    MFA_rec_2                          Reconstruction des vecteurs observés après clustering
                                       puis RCA sur les différents clusters (Adapté au modèle (M.4.2)).
    CA_graph                           Représentation graphique de la différence entre deux ensembles de vecteurs
                                       (Adapté aux modèles (M.1) et (M.2)).
    MFA_graph                          Représentation graphique de la différence entre deux ensembles de vecteurs,
                                       coloriés selon leur clusters (Adapté aux modèles (M.4) et (M.5)).
    discard                            Amputation de colonnes d'une matrice de vecteurs selon un vecteur de 0 et de 1.
    disarg                             Permutation de colonnes d'une matrice de vecteurs selon un vecteur de 0 et de 1
                                       (opération inverse de la fonction rearg).
    restit                             Ajout de colonnes vides à une matrice de vecteurs selon un vecteur de 0 et de 1.
    rearg                              Permutation de colonnes d'une matrice de vecteurs selon un vecteur de 0 et de 1
                                       (opération inverse de la fonction disarg).
    da_matrix                          Matrice de permutation associée à la permutation disarg(.,iota)
    ra_matrix                          Matrice de permutation associée à la permutation rearg(.,iota)
    FS_rec1                            Reconstruction des vecteurs observés après (P)PCA puis sélection de variables
                                       (Adapté aux modèles (M.6.1) et (M.6.3)).
    FS_rec2                            Reconstruction des vecteurs observés après RCA puis sélection de variables
                                       (Adapté aux modèles (M.6.2) et (M.6.4)).
    FS_mixrec1                         Reconstruction des vecteurs observés après clustering, (P)PCA cluster par cluster,
                                       puis sélection de variables cluster par cluster (Adapté au modèle (M.6.5)).
    FS_mixrec2                         Reconstruction des vecteurs observés après clustering, RCA cluster par cluster,
                                       puis sélection de variables cluster par cluster (Adapté au modèle (M.6.6)).
    FS_sperec1                         Reconstruction des vecteurs observés après clustering, sélection de variables,
                                       puis (P)PCA (Adapté au modèle (M.6.7)).
    FS_sperec2                         Reconstruction des vecteurs observés après clustering, sélection de variables,
                                       puis RCA (Adapté au modèle (M.6.8)).
    ================================== ==================================================================================
'''

# Version 2

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

import for_clus as ufc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions pour reconstruire ou représenter les données
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def PCA_rec(W,Z,mu):
    '''Reconstruction des vecteurs observés après (P)PCA.
    (Adapté aux modèles (M.1) et (M.5.1))
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents.
    
    mu : 1-D ndarray,
        Vecteur de taille D, supposé être la moyenne des observations.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs reconstruits.
    '''
    Y = Z @ np.transpose(W) + mu
    return Y

def RCA_rec(W,Z,V,X,mu):
    '''Reconstruction des vecteurs observés après RCA.
    (Adapté aux modèles (M.2) et (M.5.2))
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents.
    
    V : 2-D ndarray,
        Matrice de taille (D,C) d'effets fixes.
        
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables.
        L'algorithme s'assure que ces vecteurs sont centrés en les recentrant.
    
    mu : 1-D ndarray,
        Vecteur de taille D, supposé être la moyenne des observations.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs reconstruits.
    '''
    Xc = X - np.mean(X,axis=0)
    Y = Z @ np.transpose(W) + Xc @ np.transpose(V) + mu
    return Y

def MFA_rec_1(thetas,Z,omega):
    '''Reconstruction des vecteurs observés après clustering puis (P)PCA sur les différents clusters.
    (Adapté au modèle (M.4.1))
    
    Paramètres
    ----------
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [W,mu,sigma2] où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux du cluster.
            - mu est un vecteur de taille D, supposé être la moyenne des observations du cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations du cluster.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs reconstruits.
    '''
    D,L = np.shape(thetas[0][0])
    N = len(omega)
    Y = np.zeros((N,D))
    
    for n in range(N):
        W,mu,sigma2 = thetas[omega[n]]
        z = Z[n]
        Y[n] = W@z + mu
    
    return Y

def MFA_rec_2(thetas,Z,X,omega):
    '''Reconstruction des vecteurs observés après clustering puis RCA sur les différents clusters.
    (Adapté au modèle (M.4.2))
    
    Paramètres
    ----------
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [W,V,mu,sigma2] où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux du cluster.
            - V est une matrice de taille (D,C) d'effets fixes.
            - mu est un vecteur de taille D, supposé être la moyenne des observations du cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations du cluster.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables.
        L'algorithme s'assure que ces vecteurs sont centrés en les recentrant, cluster par cluster.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs reconstruits.
    '''
    D,L = np.shape(thetas[0][0])
    N = len(omega)
    Y = np.zeros((N,D))
    tri_X = ufc.tri(X,omega)
    
    for n in range(N):
        W,V,mu,sigma2 = thetas[omega[n]]
        z = Z[n]
        x = X[n] - np.mean(tri_X[omega[n]],axis=0)
        
        Y[n] = W@z + V@x + mu
    
    return Y

def CA_graph(Y,Y_hat):
    '''Représentation graphique de la différence entre deux ensembles de même nombre de vecteurs de même taille.
    (Adapté aux modèles (M.1) et (M.2))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à comparer.
    
    Y_hat : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à comparer.
    
    Renvois
    -------
    None
    '''
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
    '''Représentation graphique de la différence entre deux ensembles de même nombre de vecteurs de même taille, coloriés selon leur clusters.
    (Adapté aux modèles (M.4) et (M.5))
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à comparer.
    
    Y_hat : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à comparer.
    
    omega_hat : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu de l'ensemble de vecteurs Y_hat.
    
    omega : 1-D ndarray, optional,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu de l'ensemble de vecteurs Y.
        Si mis sur None, prend la valeur de omega_hat.
        Sinon, sera permuté pour être optimal avec omega_hat avec la fonction perm_opt avant représentation graphique.
        Mis sur None par défaut.
        
    labels : list of str, optional,
        Liste de K éléments, contenant les noms des clusters.
    
    Renvois
    -------
    None
    '''
    N,D = np.shape(Y)
    K = int(max(omega_hat) + 1)
    
    if type(omega) == type(None):
        s_omega = omega_hat
        K = int(max(omega_hat) + 1)
    else:
        s_opt = ufc.perm_opt(omega,omega_hat)
        s_omega = np.array([s_opt[k] for k in omega]).astype(int)
        K = int(max(omega_hat) + 1)
        
    tri_Y = ufc.tri(Y,s_omega,K)
    tri_Y_hat = ufc.tri(Y_hat,omega_hat,K)
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
    '''Amputation de colonnes d'une matrice de vecteurs selon un vecteur de 0 et de 1.
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à amputer.
    
    iota : 1-D ndarray,
        Vecteur de taille D rempli de 0 et de 1, où 0 signifie que la colonne est à jeter, et 1 signifie que la colonne est à garder.
        On note U le nombre de 1 dans iota.
        
    Renvois
    -------
    Y_tilde : 2-D ndarray,
        Matrice de taille (N,U) dont les colonnes ont été amputées selon iota.
    '''
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
    '''Permutation de colonnes d'une matrice de vecteurs selon un vecteur de 0 et de 1, opération inverse de la fonction rearg.
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les colonnes sont à permuter.
    
    iota : 1-D ndarray,
        Vecteur de taille D rempli de 0 et de 1, où 0 signifie que la colonne est à placer à la fin, et 1 signifie que la colonne est à placer au début.
        
    Renvois
    -------
    Y_tilde : 2-D ndarray,
        Matrice de taille (N,D) dont les colonnes ont été permutées selon iota.
    '''
    Y_v = discard(Y,iota)
    Y_u = discard(Y,1-iota)
    Y_tilde = np.concatenate([Y_v,Y_u],axis=1)
    return Y_tilde
        
def restit(Y,iota):
    '''Ajout de colonnes vides à une matrice de vecteurs selon un vecteur de 0 et de 1.
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,U) dont les lignes sont les vecteurs à amputer.
    
    iota : 1-D ndarray,
        Vecteur de taille D rempli de 0 et de 1, dont le nombre de 1 est U, où 0 signifie qu'une colonne remplie de 0 est à placer, et 1 signifie qu'une colonne de Y est à placer.
        
    Renvois
    -------
    Y_tilde : 2-D ndarray,
        Matrice de taille (N,D) dont les colonnes ont été aérées de 0 selon iota.
    '''
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
    '''Permutation de colonnes d'une matrice de vecteurs selon un vecteur de 0 et de 1, opération inverse de la fonction disarg.
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les colonnes sont à permuter.
    
    iota : 1-D ndarray,
        Vecteur de taille D rempli de 0 et de 1, dont le nombre de 1 est U, où 0 signifie que la colonne est à prendre parmi les D-U dernières, et 1 signifie que la colonne est à prendre parmi les U premières.
        
    Renvois
    -------
    Y_tilde : 2-D ndarray,
        Matrice de taille (N,D) dont les colonnes ont été permutées selon iota.
    '''
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
    '''Matrice de permutation associée à la permutation disarg(.,iota), inverse de la matrice ra_matrix(iota).
    
    Paramètres
    ----------
    iota : 1-D ndarray,
        Vecteur de taille D rempli de 0 et de 1, où 0 signifie que la colonne est à placer à la fin, et 1 signifie que la colonne est à placer au début.
        
    Renvois
    -------
    R : 2-D ndarray,
        Matrice de permutation carrée d'ordre D associée à la permutation disarg(.,iota).
    '''
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
    '''Matrice de permutation associée à la permutation rearg(.,iota), inverse de la matrice da_matrix(iota).
    
    Paramètres
    ----------
    iota : 1-D ndarray,
        Vecteur de taille D rempli de 0 et de 1, dont le nombre de 1 est U, où 0 signifie que la colonne est à prendre parmi les D-U dernières, et 1 signifie que la colonne est à prendre parmi les U premières.
        
    Renvois
    -------
    R : 2-D ndarray,
        Matrice de permutation carrée d'ordre D associée à la permutation rearg(.,iota).
    '''
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
    '''Reconstruction des vecteurs observés après (P)PCA puis sélection de variables.
    (Adapté aux modèles (M.6.1) et (M.6.3))
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice de taille (U,L) dont les colonnes sont les axes principaux.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents.
    
    mu : 1-D ndarray,
        Vecteur de taille D, supposé être la moyenne des observations.
    
    iota : 1-D ndarray,
        Vecteur de taille D, rempli de 0 et de 1, dont le nombre de 1 est U, où 0 signifie qu'une colonne remplie de 0 est à placer, et 1 signifie qu'une colonne du produit de Z par la transposée de W est à placer.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs reconstruits.
    '''
    Y_tilde = Z@np.transpose(W)
    Y_hat = restit(Y_tilde, iota) + mu
    
    return Y_hat

def FS_rec2(W,Z,V,X,mu,iota):
    '''Reconstruction des vecteurs observés après RCA puis sélection de variables.
    (Adapté aux modèles (M.6.2) et (M.6.4))
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice de taille (U,L) dont les colonnes sont les axes principaux.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents.
    
    V : 2-D ndarray,
        Matrice de taille (D,C) d'effets fixes.
        
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables.
        L'algorithme s'assure que ces vecteurs sont centrés en les recentrant.
    
    mu : 1-D ndarray,
        Vecteur de taille D, supposé être la moyenne des observations.
    
    iota : 1-D ndarray,
        Vecteur de taille D, rempli de 0 et de 1, dont le nombre de 1 est U, où 0 signifie qu'une colonne remplie de 0 est à placer, et 1 signifie qu'une colonne du produit de Z par la transposée de W est à placer.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs reconstruits.
    '''
    mu_X = np.mean(X,axis=0)
    Xc = X - mu_X
    
    Y_tilde = Z@np.transpose(W)
    Y_hat = restit(Y_tilde, iota) + Xc@np.transpose(V) + mu
    
    return Y_hat

def FS_mixrec1(thetas,Z,omega,iotas):
    '''Reconstruction des vecteurs observés après clustering, (P)PCA cluster par cluster, puis sélection de variables cluster par cluster.
    (Adapté au modèle (M.6.5))
    
    Paramètres
    ----------
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [W,mu,sigma2] où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux du cluster.
            - mu est un vecteur de taille D, supposé être la moyenne des observations du cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations du cluster.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    iotas : 2-D ndarray,
        Matrice de taille (K,D) remplie de 0 et de 1, dont chaque ligne contient U fois le nombre 1, où, pour chaque ligne, 0 signifie que, pour le cluster correspondant, une colonne remplie de 0 est à placer, et 1 signifie que, pour le cluster correspondant, une colonne du produit de Z par la transposée du W correspondant est à placer.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs reconstruits.
    '''
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
    '''Reconstruction des vecteurs observés après clustering, RCA cluster par cluster, puis sélection de variables cluster par cluster.
    (Adapté au modèle (M.6.6))
    
    Paramètres
    ----------
    thetas : list,
        Liste de K éléments, dont chaque élément est une liste de paramètres de la forme [W,V,mu,sigma2] où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux du cluster.
            - V est une matrice de taille (D,C) d'effets fixes du cluster.
            - mu est un vecteur de taille D, supposé être la moyenne des observations du cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations du cluster.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables.
        L'algorithme s'assure que ces vecteurs sont centrés en les recentrant, cluster par cluster.
        
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    iotas : 2-D ndarray,
        Matrice de taille (K,D) remplie de 0 et de 1, dont chaque ligne contient U fois le nombre 1, où, pour chaque ligne, 0 signifie que, pour le cluster correspondant, une colonne remplie de 0 est à placer, et 1 signifie que, pour le cluster correspondant, une colonne du produit de Z par la transposée du W correspondant est à placer.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs reconstruits.
    '''
    N = len(omega)
    K,D = np.shape(iotas)
    Y_hat = np.zeros((N,D))
    
    tri_X = ufc.tri(X,omega)
    
    for n in range(N):
        
        k = omega[n]
        W,V,mu,sigma2 = thetas[k]
        x = X[n] - np.mean(tri_X[k],axis=0)
        Y_tilde_n = np.array([W@Z[n]])
        Y_hat[n] = (restit(Y_tilde_n,iotas[k]))[0] + V@x + mu
    
    return Y_hat

def FS_sperec1(eta,Zv,Zu,iota):
    '''Reconstruction des vecteurs observés après clustering, sélection de variables, puis (P)PCA.
    (Adapté au modèle (M.6.7))
    
    Paramètres
    ----------
    eta : list,
        Liste de paramètres de la forme [Wv,Wu,mu,nu,sigma2,tau2] où :
            - Wv est une matrice de taille (Dv,Lv) dont les colonnes sont les axes principaux sur lesquels la loi des variables aléatoires change en fonction du cluster.
            - Wu est une matrice de taille (Du,Lu) dont les colonnes sont les axes principaux sur lesquels la loi des variables aléatoires ne change pas en fonction du cluster.
            - mu est un vecteur de taille Dv+Du, supposé être la moyenne des vecteurs observés.
            - nu est un vecteur de taille Lu, supposé être la moyenne des vecteurs latents dont la loi ne change pas en fonction du cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations.
            - tau2 est un réel positif, supposé être la variance des vecteurs latents dont la loi ne change pas en fonction du cluster.
    
    Zv : 2-D ndarray,
        Matrice de taille (N,Lv) dont les lignes sont les vecteurs latents dont la loi change en fonction du cluster.
    
    Zu : 2-D ndarray,
        Matrice de taille (N,Lu) dont les lignes sont les vecteurs latents dont la loi ne change pas en fonction du cluster.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables.
        L'algorithme s'assure que ces vecteurs sont centrés en les recentrant, cluster par cluster.
    
    iota : 1-D ndarray,
        Vecteur de taille Dv+Du rempli de 0 et de 1, dont le nombre de 1 est Dv, où 0 signifie que la colonne est à prendre parmi celles du produit de Zu avec la transposée de Wu, et 1 signifie que la colonne est à prendre parmi celles du produit de Zv avec la transposée de Wv.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,Dv+Du) dont les lignes sont les vecteurs reconstruits.
    '''
    Wv,Wu,mu,nu,sigma2,tau2 = eta
    
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
        
def FS_sperec2(eta,Zv,Zu,X,iota):
    '''Reconstruction des vecteurs observés après clustering, sélection de variables, puis (P)PCA.
    (Adapté au modèle (M.6.8))
    
    Paramètres
    ----------
    eta : list,
        Liste de paramètres de la forme [Wv,Wu,V,mu,nu,sigma2,tau2] où :
            - Wv est une matrice de taille (Dv,Lv) dont les colonnes sont les axes principaux sur lesquels la loi des variables aléatoires change en fonction du cluster.
            - Wu est une matrice de taille (Du,Lu) dont les colonnes sont les axes principaux sur lesquels la loi des variables aléatoires ne change pas en fonction du cluster.
            - V est une matrice de taille (Dv+Du,C) d'effets fixes.
            - mu est un vecteur de taille Dv+Du, supposé être la moyenne des vecteurs observés.
            - nu est un vecteur de taille Lu, supposé être la moyenne des vecteurs latents dont la loi ne change pas en fonction du cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations.
            - tau2 est un réel positif, supposé être la variance des vecteurs latents dont la loi ne change pas en fonction du cluster.
    
    Zv : 2-D ndarray,
        Matrice de taille (N,Lv) dont les lignes sont les vecteurs latents dont la loi change en fonction du cluster.
    
    Zu : 2-D ndarray,
        Matrice de taille (N,Lu) dont les lignes sont les vecteurs latents dont la loi ne change pas en fonction du cluster.
    
    iota : 1-D ndarray,
        Vecteur de taille Dv+Du rempli de 0 et de 1, dont le nombre de 1 est Dv, où 0 signifie que la colonne est à prendre parmi les Du dernières, et 1 signifie que la colonne est à prendre parmi les Dv premières.
        
    Renvois
    -------
    Y : 2-D ndarray,
        Matrice de taille (N,Dv+Du) dont les lignes sont les vecteurs reconstruits.
    '''
    Wv,Wu,V,mu,nu,sigma2,tau2 = eta
    
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