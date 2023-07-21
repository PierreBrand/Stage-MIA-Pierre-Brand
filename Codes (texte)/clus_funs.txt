'''=======================
    Fonctions de clustering
    =======================
    
    ================================== ===================================================================
    Contient les fonctions suivantes :
    ---------------------------------- -------------------------------------------------------------------
    HAC_SL                             Clustering Agglomératif Hiérarchique, en distance Single-Linkage.
    HAC_CL                             Clustering Agglomératif Hiérarchique, en distance Complete-Linkage.
    HAC_AL                             Clustering Agglomératif Hiérarchique, en distance Average-Linkage.
    HAC_Ward                           Clustering Agglomératif Hiérarchique, en distance de Ward.
    K_means                            Algorithme itératif K-means, ou "K-moyennes".
    K_means_FPC                        Algorithme itératif K-means ++.
    K_medoids                          Algorithme itératif K-medoids.
    Lapras                             Clustering Spectral.
    K_opt                              Estimation du nombre de clusters.
    ================================== ===================================================================
'''

# Version 3

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions de clustering
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def HAC_SL(X,K,tempo=False):
    '''Clustering Agglomératif Hiérarchique, en distance Single-Linkage.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    K : int,
        Nombre de clusters à former.
    
    tempo : bool, optional,
        Ne sert à rien, quelle que soit sa valeur.
        Mis sur False par défaut.
    
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    clusters = sklclu.AgglomerativeClustering(K,linkage='single').fit(X)
    omega = clusters.labels_
    
    return omega

def HAC_CL(X,K,tempo=False):
    '''Clustering Agglomératif Hiérarchique, en distance Complete-Linkage.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    K : int,
        Nombre de clusters à former.
    
    tempo : bool, optional,
        Ne sert à rien, quelle que soit sa valeur.
        Mis sur False par défaut.
    
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    clusters = sklclu.AgglomerativeClustering(K,linkage='complete').fit(X)
    omega = clusters.labels_
    
    return omega

def HAC_AL(X,K,tempo=False):
    '''Clustering Agglomératif Hiérarchique, en distance Average-Linkage.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    K : int,
        Nombre de clusters à former.
    
    tempo : bool, optional,
        Ne sert à rien, quelle que soit sa valeur.
        Mis sur False par défaut.
    
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    clusters = sklclu.AgglomerativeClustering(K,linkage='average').fit(X)
    omega = clusters.labels_
    
    return omega

def HAC_Ward(X,K,tempo=False):
    '''Clustering Agglomératif Hiérarchique, en distance de Ward.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    K : int,
        Nombre de clusters à former.
    
    tempo : bool, optional,
        Ne sert à rien, quelle que soit sa valeur.
        Mis sur False par défaut.
    
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    clusters = sklclu.AgglomerativeClustering(K,linkage='ward').fit(X)
    omega = clusters.labels_
    
    return omega

def K_means(X,omega,nb_steps=100,tempo=True):
    '''Algorithme itératif K-means, ou "K-moyennes".
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme K-means.
        Mis sur 100 par défaut.
        L'algorithme s'arrête de lui-même si jamais la configuration obtenue est déjà stable.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme.
        Mis sur True par défaut.
        
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    N,D = np.shape(X)
    occ = ufc.occurences(omega)
    K = len(occ)
    if np.any(occ==0) :
        print("K_means : Clusters vides")
    
    omega_hat = omega.copy()
    t = 0
    dist = 1
    M = np.zeros((K,D))
    
    while dist > 0 and t < nb_steps :
        
        #Recalcul des centres
        tri_X = ufc.tri(X,omega_hat,K)
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
    '''Algorithme K-means ++, i.e. K-means mais qui s'initialise en prenant comme centres des clusters des vecteurs du jeu de données X.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    K : int,
        Nombre de clusters à former.
    
    determ : bool, optional,
        Si mis sur True, l'algorithme s'initialise en prenant comme centres des deux premiers clusters les deux vecteurs les plus éloignés, puis prend successivement, comme centres des autres clusters, les vecteurs dont la distance L2 minimale aux vecteurs déjà sélectionnés est maximale. 
        Si mis sur False, l'algorithme s'initialise en prenant comme centre du premier cluster un vecteur uniformément au hasard, puis prend successivement, comme centres des autres clusters, des vecteurs au hasard, chacun pondérés proportionnellement à leur distance L2 minimale aux vecteurs déjà sélectionnés.
        Mis sur True par défaut.
        
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme K-means.
        Mis sur 100 par défaut.
        L'algorithme s'arrête de lui-même si jamais la configuration obtenue est déjà stable.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
        
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    N = len(X)
    if N < K :
        print("N < K")
    else :
        
        #Initialisation
        
        if determ :
            Pdm = ufc.dissim_L2(X)
            ind_max = np.argmax(Pdm)
            n0 = int(ind_max/N)
            n1 = ind_max%N
            
            M = np.array([X[n0],X[n1]])
            
            for k in range(2,K):
            
                dists_CP = np.array([ufc.Dist_CP(x,M) for x in X])
                n_k = np.argmax(dists_CP)
                
                M_list = list(M)
                M_list.append(X[n_k])
                M = np.array(M_list)
        
        else :
            n0 = rd.choice(N)
            M = np.array([X[n0]])
            
            for k in range(1,K):
                
                dists_CP = np.array([ufc.Dist_CP(x,M) for x in X])
                
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
    '''Algorithme itératif K-medoids, ou "K-médoïdes", ou encore "K-centroïdes" (ew).
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme K-means.
        Mis sur 100 par défaut.
        L'algorithme s'arrête de lui-même si jamais la configuration obtenue est déjà stable.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme.
        Mis sur True par défaut.
        
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    N,D = np.shape(X)
    occ = ufc.occurences(omega)
    K = len(occ)
    if np.any(occ==0) :
        print("K_means : Clusters vides")
    
    omega_hat = omega.copy()
    t = 0
    dist = 1
    M = np.zeros((K,D))
    
    while dist > 0 and t < nb_steps :
        
        #Recalcul des médoïdes
        tri_X = ufc.tri(X,omega_hat,K)
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

def Lapras(X,K,determ=True,nb_steps=100,tempo=True):
    '''Clustering Spectral utilisant, pour matrice de similarité, la matrice de dissimilarité donnée par dissim_L2 renormalisée, dont chaque coefficient est passé par la fonction inverse de l'exponentielle.
        (La fonction s'appelle Lapras parce que c'est le nom en anglais d'un Pokémon dont le nom en japonais est Laplace.)
        
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    K : int,
        Nombre de clusters à former.
    
    determ : bool, optional,
        Si mis sur True, l'algorithme K-means++ s'initialisera en prenant comme centres des deux premiers clusters les deux vecteurs les plus éloignés, puis prend successivement, comme centres des autres clusters, les vecteurs dont la distance L2 minimale aux vecteurs déjà sélectionnés est maximale. 
        Si mis sur False, l'algorithme K-means++ s'initialisera en prenant comme centre du premier cluster un vecteur uniformément au hasard, puis prend successivement, comme centres des autres clusters, des vecteurs au hasard, chacun pondérés proportionnellement à leur distance L2 minimale aux vecteurs déjà sélectionnés.
        Mis sur True par défaut.
        
    nb_steps : int, optional,
        Nombre maximal d'itérations de l'algorithme K-means.
        Mis sur 100 par défaut.
        L'algorithme s'arrête de lui-même si jamais la configuration obtenue est déjà stable.
        
    tempo : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le nombre d'itérations effectuées par l'algorithme K-means.
        Mis sur True par défaut.
        
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    N,D = np.shape(X)
    
    #Initialisation
    Pdm = ufc.dissim_L2(X)
    alpha = np.mean(Pdm)
    Psm = np.exp(-N/(N-1)/alpha*Pdm)
    Lokh = ufc.Lap(Psm)

    #EVP
    SpL,P = nla.eig(Lokh)
    P2 = mbu.normalize(P)
    ordre = np.sort(SpL)
    inlist = [k for k in range(N) if SpL[k] in ordre[:K]]
    tU = np.array([P2[:,k] for k in inlist])
    T = np.transpose(mbu.normalize(tU))
    omega = K_means_FPC(T,K,determ=determ,nb_steps=nb_steps,tempo=tempo)
    
    return omega

#K optimal
def K_opt(X,K_min=2,alpha=None,detail=False):
    '''Estimation du nombre de clusters à former à partir du spectre de la matrice de dissimilarité donnée par la fonction dissim_L2.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs à clusteriser.
    
    K_min : int, optional,
        Nombre minimal de clusters à dégager.
    
    alpha : float, optional,
        Coefficient de renormalisation de la matrice de dissimilarité donnée par la fonction dissim_L2.
        Si mis sur None, prendra comme valeur l'inverse de la moyenne des distances L2 entre les vecteurs.
    
    detail : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le détail graphique de l'estimation du nombre de clusters.
        Mis sur False par défaut.
        
    Renvois
    -------
    K : int,
        Estimation du nombre optimal de clusters à partir du plus grand saut dans le log de l'opposé du spectre de la la matrice de dissimilarité donnée par la fonction dissim_L2, renormalisée.
    '''
    N,D = np.shape(X)    
    Pdm = ufc.dissim_L2(X)
    
    if type(alpha) == type(None):
        alpha = 1.0/np.mean(Pdm)
    
    norm_Pdm = alpha*Pdm
    SpP,Q = nla.eig(norm_Pdm)
    ordre = np.sort(SpP)[:-1]
    
    logm_ordre = np.log(-ordre)
    diff = logm_ordre[:-1]-logm_ordre[1:]
    
    if K_min >= 2:
        diff[:K_min-2] = [0]*(K_min-2)
    K_star = int(np.argmax(diff)) + 2
    
    if detail:
        
        plt.figure()
        plt.step(np.arange(2,N+1),logm_ordre)
        plt.plot([K_star,K_star],[logm_ordre[0],logm_ordre[-1]],'--',label='$K_{star}$')
        plt.title("Log de l'opposé du spectre de la matrice de dissimilarité")
        plt.legend()
        plt.figure()
    
    return K_star