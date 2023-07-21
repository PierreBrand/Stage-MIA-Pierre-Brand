'''===============================================
    Fonctions utiles pour la sélection de variables
    ===============================================
    
    ================================== ==========================================================================
    Contient les fonctions suivantes :
    ---------------------------------- --------------------------------------------------------------------------
    cor_emp                            Matrice de corrélation empirique des matrices de vecteurs.
    FWIR                               "Feature-Wise Inertia Rate" ou "Part d'inertie, variable par variable".
    U_opt                              Nombre optimal de dimensions à conserver, estimé par la "méthode du saut".
    iotate                             Estimation du vecteur induisant une sélection de variables
                                       (comparaison des parts d'inerties, variable par variable).
    iotate_2                           Estimation du vecteur induisant une sélection de variables
                                       (comparaison des coefficients de corrélation).
    ================================== ==========================================================================
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
#Fonctions utiles pour la sélection de variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def cor_emp(Y,Z):
    '''Matrice de corrélation empirique des matrices de vecteurs Y et Z.
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D1) dont les lignes sont des réalisations supposément indépendantes de vecteurs aléatoires de taille D.
    
    Z : 2-D ndarray,
        Matrice de taille (N,D2) dont les lignes sont des réalisations supposément indépendantes de vecteurs aléatoires de taille L.
        
    Renvois
    -------
    Cor : 2-D ndarray,
        Matrice de taille (D2,D1) dont le coefficient pour chaque ligne et chaque colonne est le coefficient de corrélation entre la ligne de Z et la colonne de Y correspondantes.
    '''
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

def FWIR(X,omega):
    '''"Feature-Wise Inertia Rate" ou "Part d'inertie, variable par variable".
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs appartenant à différents clusters.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
       
    Renvois
    -------
    I_X : 2-D ndarray,
        Matrice de taille (N,D) dont le coefficient est la part d'inertie de l'individu en ligne selon l'axe en colonne.
    '''
    N,D = np.shape(X)
    N1 = len(omega)
    
    if N != N1:
        print("X et omega n'ont pas le même nombre d'individus")
    else :
        occ = ufc.occurences(omega)
        K = len(occ)
        if not np.all(occ):
            print("Au moins un des clusters est vide")
        else:

            tri_X = ufc.tri(X,omega)
            M = np.array([np.mean(tri_X[k],axis=0) for k in range(K)])
            mu_glob = np.mean(X,axis=0)

            I_loc = np.array([[(X[n][d] - M[omega[n]][d])**2 for d in range(D)] for n in range(N)])
            I_glob = np.array([[(X[n][d] - mu_glob[d])**2 for d in range(D)] for n in range(N)])
            I_X = I_glob/I_loc

            return I_X

def U_opt(contrib,U_min=1,detail=False):
    '''Nombre optimal de dimensions à conserver, estimé par la "méthode du saut".
    
    Paramètres
    ----------
    contrib : 1-D ndarray,
        Vecteur de taille D dont les coefficients sont d'autant plus grands que la dimension correspondante est utile pour le clustering.
    
    U_min : int, optional,
        Nombre minimal de variables à conserver.
        Mis sur 1 par défaut.
        
    detail : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le détail graphique de l'estimation du nombre de dimensions utiles.
        Mis sur False par défaut.
    
    Renvois
    -------
    U : int,
        Nombre optimal de dimensions à conserver.
    '''
    D = len(contrib)
    
    if U_min < 1:
        U_min = 1
    
    if D <= U_min:
        print("D inférieur ou égal à U_min")
    else:
        ordre = np.sort(contrib)
        diff_ordre = np.concatenate([[0],ordre[1:D-U_min+1] - ordre[:D-U_min]])
        U_star = D - int(np.argmax(diff_ordre))
        
        if detail:
            
            plt.figure()
            plt.step(np.arange(0,D),ordre,label='Contribution')
            plt.plot([D-U_star-1,D-U_star-1],[ordre[D-U_star-1],ordre[D-U_star]])
            plt.plot([D-U_star-1,D-U_star-1],[ordre[D-U_star-1],0],'--',label='$U_{star}$')
            plt.legend()
            plt.show()
        
        return U_star

def iotate(X,omega,U=None,detail=False):
    '''Estimation du vecteur iota induisant une sélection de variables, en comparant les parts d'inerties, variable par variable.
    
    Paramètres
    ----------
    X : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs appartenant à différents clusters.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    U : int, optional,
        Nombre de variables à conserver.
        Si mis sur None, le nombre de variables à conserver est estimé à partir de la fonction U_opt.
        Mis sur None par défaut.
        
    detail : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le détail graphique de la sélection de variables.
        Mis sur False par défaut.
    
    Renvois
    -------
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est à conserver, et le nombre 0 signifie qu'elle ne l'est pas.
    '''
    N,D = np.shape(X)
    N1 = len(omega)
    
    if N != N1:
        print("X et omega n'ont pas le même nombre d'individus")
    else :
        occ = ufc.occurences(omega)
        K = len(occ)
        if not np.all(occ):
            print("Au moins un des clusters est vide")
        else:
            I_X = FWIR(X,omega)
            contrib = np.sum(I_X,axis=0)
            ordre = np.sort(contrib)

            if type(U) == type(None):
                U = U_opt(contrib,detail=detail)

            iota = np.array([int(contrib[d] in ordre[D-U:]) for d in range(D)])

            return iota

def iotate_2(Y,Z,U=None,dist=2,detail=False):
    '''Estimation du vecteur iota induisant une sélection de variables, en comparant les coefficients de corrélation, variable par variable.
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont des vecteurs observés.
    
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont des vecteurs latents dont la loi change selon les clusters.
    
    U : int, optional,
        Nombre de variables à conserver.
        Si mis sur None, le nombre de variables à conserver est estimé à partir de la fonction U_opt.
        Mis sur None par défaut.
    
    dist = int, float, or str, optional,
        Norme utilisée pour mesurer les colonnes de la matrice de corrélation empirique.
    
    detail : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche le détail graphique de la sélection de variables.
        Mis sur False par défaut.
    
    Renvois
    -------
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est à conserver, et le nombre 0 signifie qu'elle ne l'est pas.
    '''
    N1,D = np.shape(Y)
    N2,L = np.shape(Z)
    
    if N1 != N2:
        print("Y et Z sont de tailles différentes")
    else :
        N = N1
    
        Cor = cor_emp(Y,Z)
        
        if (type(dist) == int or type(dist) == float) and dist >= 1:
            contrib = (np.sum(np.abs(Cor)**dist,axis=0))**(1/dist)
        else :
            if dist == 'inf':
                contrib = np.max(np.abs(Cor),axis=0)
            else :
                print("Distance non-reconnue.")
            
            
        ordre = np.sort(contrib)
        
        if type(U) == type(None):
            U = U_opt(ordre,L,detail)
        
        brink = ordre[D-U]
        iota_hat = (contrib>=brink).astype(int)
        
        return iota_hat