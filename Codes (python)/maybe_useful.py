'''==========================
    Fonctions peut-être utiles
    ==========================
    
    ================================== ====================================================================
    Contient les fonctions suivantes :
    ---------------------------------- --------------------------------------------------------------------
    norme                              Norme L2 d'un vecteur.
    angle_oriente                      Angle orienté entre un vecteur de taille 2 et le vecteur (1,0).
    rotation                           Matrice de rotation de taille (2,2) associée à un angle.
    erreur_Z                           Distance en norme L2 entre deux matrices à 2 colonnes,
                                       après avoir rotaté l'une des deux.
    orthogonalize                      Orthogonalisation d'une matrice injective.
    normalize                          Normalisation des colonnes d'une matrice.
    trace                              Trace d'une matrice carrée.
    cov_emp                            Matrice de covariance empirique de vecteurs aléatoires i.i.d.
    L_knee                             Emplacement du "genou" d'un vecteur, si celui-ci se trouve à gauche.
    R_knee                             Emplacement du "genou" d'un vecteur, si celui-ci se trouve à droite.
    R_elbow                            Emplacement du "coude" d'un vecteur, si celui-ci se trouve à droite.
    L_elbow                            Emplacement du "coude" d'un vecteur, si celui-ci se trouve à gauche.
    ================================== ====================================================================
'''

#Version 2

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
    '''Norme L2 du vecteur x.
    
    Paramètres
    ----------
    x : 1-D ndarray,
        Vecteur dont on veut calculer la norme.
    
    Renvois
    -------
    norme_x : float,
        Norme L2 du vecteur x.
    '''
    return np.sqrt(x@x)

def angle_oriente(x):
    '''Angle orienté entre vecteur x et le vecteur (1,0).
    
    Paramètres
    ----------
    x : 1-D ndarray,
        Vecteur non-nul de longueur 2 dont on veut calculer l'angle orienté avec le vecteur (1,0).
    
    Renvois
    -------
    alpha : float,
        Angle orienté entre vecteur x et le vecteur (1,0).
    '''
    
    nx = x/norme(x)
    if nx[1] >= 0:
        return np.arccos(nx[0])
    else :
        return -np.arccos(nx[0])

def rotation(a):
    '''Matrice de rotation de taille (2,2) associée à l'angle a.
    
    Paramètres
    ----------
    a : float,
        Angle dont on veut la matrice de rotation de taille (2,2) associée.
    
    Renvois
    -------
    R : 2-D ndarray,
        Matrice de rotation de taille (2,2) associée à l'angle a.
    '''
    R = np.array([[np.cos(a), np.sin(a)],[-np.sin(a), np.cos(a)]])
    return R

def erreur_Z(Z1,Z2,a,sym=False):
    '''Distance en norme L2 entre la matrice Z1, et le produit de la matrice de rotation associée à l'angle a par la matrice Z2.
    
    Paramètres
    ----------
    Z1 : 2-D ndarray,
        Matrice de taille (N,2).
    
    Z2 : 2-D ndarray,
        Matrice de taille (N,2).
        
    a : float,
        Angle avec lequel on veut rotater les lignes de la matrices Z2.
    
    sym : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, reflète orthogonalement les vecteurs de Z2 sur l'axe des y avant de les rotater.
        Mis sur False par défaut.
        
    Renvois
    -------
    dist : float,
        Distance en norme L2 entre la matrice Z1, et le produit de la matrice de rotation associée à l'angle a par la matrice Z2.
    '''
    S = np.array([[1,0],[0,1-2*float(sym)]])
    R = np.transpose(rotation(a)@S)
    RZ2 = Z2@R
    return np.sum((Z1-RZ2)**2)

def orthogonalize(W):
    '''Matrice de même taille que W dont les colonnes sont orthogonales.
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice injective à orthogonaliser.
    
    Renvois
    -------
    W_orth : 2-D ndarray,
        Matrice de même taille que W dont les colonnes sont orthogonales.
    '''
    R2 = np.transpose(W)@W
    SpR2, P = nla.eig(R2)
    R = np.real(P @ np.diag(1/np.sqrt(SpR2)) @ nla.inv(P))
    W_2 = W @ R
    return W_2

def normalize(P):
    '''Matrice de même ensemble d'images que P, mais dont les colonnes sont de norme 1.
    
    Paramètres
    ----------
    P : 2-D ndarray,
        Matrice injective à normaliser.
    
    Renvois
    -------
    P_orth : 2-D ndarray,
        Matrice de même ensemble d'images que P, mais dont les colonnes sont de norme 1.
    '''
    P_norms = np.sqrt(np.sum(P**2,axis=0))
    D_P = np.diag(1/P_norms)
    P2 = P@D_P
    return P2

def trace(A):
    '''Trace de la matrice A.
    
    Paramètres
    ----------
    A : 2-D ndarray,
        Matrice carrée dont on veut calculer la trace.
    
    Renvois
    -------
    tr_A : float,
        Trace de la matrice A.
    '''
    N1,N2 = np.shape(A)
    if N1 != N2 :
        print('Erreur de dimensions sur A')
    else :
        return np.sum([A[n][n] for n in range(N1)])

def cov_emp(Y):
    '''Matrice de covariance empirique des vecteurs de la matrice Y.
    
    Paramètres
    ----------
    Y : 2-D ndarray,
        Matrice dont les lignes sont des vecteurs aléatoires, supposément indépendants et de même loi, dont on veut calculer la covariance.
    
    Renvois
    -------
    S : 2-D ndarray,
        Matrice de covariance empirique des vecteurs de la matrice Y.
    '''
    N,D = np.shape(Y)
    mu_Y = np.mean(Y,axis=0)
    Yc = Y - mu_Y
    S = 1/N * np.transpose(Yc) @ Yc
    return S

def L_knee(y,x=None,alpha=1.0):
    '''Emplacement du "genou" du vecteur y, si celui-ci se trouve à gauche.
    
    Paramètres
    ----------
    y : 1-D ndarray,
        Vecteur d'ordonnées dont on veut calculer l'emplacement du "genou".
    
    x : 1-D ndarray, optional,
        Vecteur d'ordonnées dont on veut calculer l'emplacement du "genou".
        Mis par défaut sur np.range(0,N) ou N est la taille de y.
    
    alpha : float, optional,
        Coefficient pour ajuster la part d'importance des abscisses et des ordonnées dans le calcul de l'emplacement du "genou".
        Plus alpha est grand, plus la part d'importance des ordonnées dans le calcul de l'emplacement du "genou" sera grande, et donc plus le "genou" sera pris à droite.
        Mis sur 1.0 par défaut.
        
    Renvois
    -------
    n_knee : int,
        Emplacement du "genou" du vecteur y, si celui-ci se trouve à gauche.
    '''
    N = len(y)
    ord_y = np.sort(y)
    
    if type(x) == type(None) or len(x) != N:
        x = np.arange(0,N)
    ord_x = np.sort(x)
    
    #Renormalisation
    vert_length = ord_y[-1]-ord_y[0]
    horz_length = ord_x[-1]-ord_x[0]
    x1_list = np.array([(x-ord_x[0])/horz_length for x in ord_x])
    y1_list = np.array([(y-ord_y[0])/vert_length for y in ord_y])
    z1_list = alpha*y1_list - x1_list
    
    n_star = int(np.argmax(z1_list))
    return n_star

def R_knee(y,x=None,alpha=1.0):
    '''Emplacement du "genou" du vecteur y, si celui-ci se trouve à droite.
    
    Paramètres
    ----------
    y : 1-D ndarray,
        Vecteur d'ordonnées dont on veut calculer l'emplacement du "genou".
    
    x : 1-D ndarray, optional,
        Vecteur d'ordonnées dont on veut calculer l'emplacement du "genou".
        Mis par défaut sur np.range(0,N) ou N est la taille de y.
    
    alpha : float, optional,
        Coefficient pour ajuster la part d'importance des abscisses et des ordonnées dans le calcul de l'emplacement du "genou".
        Plus alpha est grand, plus la part d'importance des ordonnées dans le calcul de l'emplacement du "genou" sera grande, et donc plus le "genou" sera pris à gauche.
        Mis sur 1.0 par défaut.
        
    Renvois
    -------
    n_knee : int,
        Emplacement du "genou" du vecteur y, si celui-ci se trouve à droite.
    '''    
    N = len(y)
    return N - L_knee(y,x,alpha)

def R_elbow(y,x=None,alpha=1.0):
    '''Emplacement du "coude" du vecteur y, si celui-ci se trouve à droite.
    
    Paramètres
    ----------
    y : 1-D ndarray,
        Vecteur d'ordonnées dont on veut calculer l'emplacement du "coude".
    
    x : 1-D ndarray, optional,
        Vecteur d'ordonnées dont on veut calculer l'emplacement du "coude".
        Mis par défaut sur np.range(0,N) ou N est la taille de y.
    
    alpha : float, optional,
        Coefficient pour ajuster la part d'importance des abscisses et des ordonnées dans le calcul de l'emplacement du "coude".
        Plus alpha est grand, plus la part d'importance des ordonnées dans le calcul de l'emplacement du "coude" sera grande, et donc plus le "coude" sera pris à gauche.
        Mis sur 1.0 par défaut.
        
    Renvois
    -------
    n_elbow : int,
        Emplacement du "coude" du vecteur y, si celui-ci se trouve à droite.
    '''
    N = len(y)
    ord_y = np.sort(y)
    
    if type(x) == type(None) or len(x) != N:
        x = np.arange(0,N)
    ord_x = np.sort(x)
    
    #Renormalisation
    vert_length = ord_y[-1]-ord_y[0]
    horz_length = ord_x[-1]-ord_x[0]
    x1_list = np.array([(x-ord_x[0])/horz_length for x in ord_x])
    y1_list = np.array([(y-ord_y[0])/vert_length for y in ord_y])
    z1_list = x1_list - alpha*y1_list
    
    n_star = int(np.argmax(z1_list))
    return n_star

def L_elbow(y,x=None,alpha=1.0):
    '''Emplacement du "coude" du vecteur y, si celui-ci se trouve à gauche.
    
    Paramètres
    ----------
    y : 1-D ndarray,
        Vecteur d'ordonnées dont on veut calculer l'emplacement du "coude".
    
    x : 1-D ndarray, optional,
        Vecteur d'ordonnées dont on veut calculer l'emplacement du "coude".
        Mis par défaut sur np.range(0,N) ou N est la taille de y.
    
    alpha : float, optional,
        Coefficient pour ajuster la part d'importance des abscisses et des ordonnées dans le calcul de l'emplacement du "coude".
        Plus alpha est grand, plus la part d'importance des ordonnées dans le calcul de l'emplacement du "coude" sera grande, et donc plus le "coude" sera pris à droite.
        Mis sur 1.0 par défaut.
        
    Renvois
    -------
    n_elbow : int,
        Emplacement du "coude" du vecteur y, si celui-ci se trouve à gauche.
    '''
    N = len(y)
    return N - R_elbow(y,x,alpha)