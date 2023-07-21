'''====================================================
    Fonctions pour simuler des paramètres et des données
    ====================================================
    
    ================================== ======================================================================================
    Contient les fonctions suivantes :
    ---------------------------------- --------------------------------------------------------------------------------------
    param_lin                          Ensemble de paramètres pour le modèle linéaire Gaussien simple (Modèle (M.1)).
    param_cov                          Ensemble de paramètres pour le modèle linéaire Gaussien mixte (Modèle (M.2)).
    param_obsmix_1                     Ensembles de paramètres pour la mixture sur l'espace observé
                                       de modèles linéaires Gaussiens simples (Modèle (M.4.1)).
    param_obsmix_2                     Ensembles de paramètres pour la mixture sur l'espace observé
                                       de modèles linéaires Gaussiens mixtes (Modèle (M.4.2)).
    param_latmix_1                     Ensembles de paramètres pour la mixture sur l'espace latent
                                       de modèles linéaires Gaussiens simples (Modèle (M.5.1)).
    param_latmix_2                     Ensembles de paramètres pour la mixture sur l'espace latent
                                       de modèles linéaires Gaussiens mixtes (Modèle (M.5.2)).
    noisy_param_1                      Ensemble de paramètres pour le modèle linéaire Gaussien simple
                                       avec variables impertinentes (Modèle (M.6.1)).
    noisy_param_2                      Ensemble de paramètres pour le modèle linéaire Gaussien mixte
                                       avec variables impertinentes (Modèle (M.6.2)).
    noisy_param_latmix_1               Ensembles de paramètres pour la mixture sur l'espace latent
                                       de modèles linéaires Gaussiens simples, avec variables impertinentes (Modèle (M.6.3)).
    noisy_param_latmix_2               Ensembles de paramètres pour la mixture sur l'espace latent
                                       de modèles linéaires Gaussiens mixtes, avec variables impertinentes (Modèle (M.6.4)).
    noisy_param_obsmix_1               Ensembles de paramètres pour la mixture sur l'espace observé
                                       de modèles linéaires Gaussiens simples, avec variables impertinentes (Modèle (M.6.5)).
    noisy_param_obsmix_2               Ensembles de paramètres pour la mixture sur l'espace observé
                                       de modèles linéaires Gaussiens mixtes, avec variables impertinentes (Modèle (M.6.6)).
    noisy_param_spemix_1               Ensembles de paramètres pour la mixture sur seulement certaines dimensions des
                                       espaces observé et latent de modèles linéaires Gaussiens simples (Modèle (M.6.7)).
    noisy_param_spemix_2               Ensembles de paramètres pour la mixture sur seulement certaines dimensions des
                                       espaces observé et latent de modèles linéaires Gaussiens mixtes (Modèle (M.6.8)).
    
    sim_omega                          Vecteur aléatoire de taille rempli d'entiers positifs.
    
    data_lin                           Ensemble de données simulées pour le modèle linéaire Gaussien simple (Modèle (M.1)).
    data_cov                           Ensemble de données simulées pour le modèle linéaire Gaussien mixte (Modèle (M.2)).
    data_obsmix_1                      Ensemble de données simulées pour la mixture sur l'espace observé
                                       de modèles linéaires Gaussiens simples (Modèle (M.4.1)).
    data_obsmix_2                      Ensemble de données simulées pour la mixture sur l'espace observé
                                       de modèles linéaires Gaussiens mixtes (Modèle (M.4.2)).
    data_latmix_1                      Ensemble de données simulées pour la mixture sur l'espace latent
                                       de modèles linéaires Gaussiens simples (Modèle (M.5.1)).
    data_latmix_2                      Ensemble de données simulées pour la mixture sur l'espace latent
                                       de modèles linéaires Gaussiens mixtes (Modèle (M.5.2)).
    noisy_data_1                       Ensemble de données simulées pour le modèle linéaire Gaussien simple
                                       avec variables impertinentes (Modèle (M.6.1)).
    noisy_data_2                       Ensemble de données simulées pour le modèle linéaire Gaussien mixte
                                       avec variables impertinentes (Modèle (M.6.2)).
    noisy_data_latmix_1                Ensemble de données simulées pour la mixture sur l'espace latent
                                       de modèles linéaires Gaussiens simples, avec variables impertinentes (Modèle (M.6.3)).
    noisy_data_latmix_2                Ensemble de données simulées pour la mixture sur l'espace latent
                                       de modèles linéaires Gaussiens mixtes, avec variables impertinentes (Modèle (M.6.4)).
    noisy_data_obsmix_1                Ensemble de données simulées pour la mixture sur l'espace observé
                                       de modèles linéaires Gaussiens simples, avec variables impertinentes (Modèle (M.6.5)).
    noisy_data_obsmix_2                Ensemble de données simulées pour la mixture sur l'espace observé
                                       de modèles linéaires Gaussiens mixtes, avec variables impertinentes (Modèle (M.6.6)).
    noisy_data_spemix_1                Ensemble de données simulées pour la mixture sur seulement certaines dimensions des
                                       espaces observé et latent de modèles linéaires Gaussiens simples (Modèle (M.6.7)).
    noisy_data_spemix_2                Ensemble de données simulées pour la mixture sur seulement certaines dimensions des
                                       espaces observé et latent de modèles linéaires Gaussiens mixtes (Modèle (M.6.8)).
    ================================== ======================================================================================
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

import maybe_useful as mbu
import data_rnr as drr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Fonctions pour simuler les données
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def param_lin(D,L,m1=0.0,m2=0.0,m3=0.0,s1=1.0,s2=1.0,s3=1.0,disp=False,orthog=True):
    '''Simule un ensemble de paramètres pour le modèle linéaire Gaussien simple.
    (Modèle (M.1))
    
    Paramètres
    ----------
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D.
    
    m1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 0.0 par défaut.
    
    s1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux seront les colonnes d'une matrice de taille (D,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,s1²)
        Si mis sur True, les axes principaux seront les colonnes d'une matrice de taille (D,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,1.0), passée par la fonction orthogonalize, puis multipliée par s1.
        Mis sur True par défaut.
        
    Renvois
    -------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux.
        
    mu : 1-D ndarray,
        Vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2,s2²).
    
    sigma2 : float,
        Réel positif obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3,s3²).
    '''
    mu = rd.normal(m2,s2**2,D)
    sigma2 = rd.normal(m3,s3**2)**2
    
    if orthog:
        W = rd.normal(m1,1.0,(D,L))
        W = s1*mbu.orthogonalize(W)
    else:
        W = rd.normal(m1,s1**2,(D,L))
    
    if disp:
        print('$W = $', W)
        print('$\mu = $', mu)
        print('$\sigma^2 = $', sigma2)
        
    return W, mu, sigma2

def data_lin(W,mu,sigma2,N,disp=False):
    '''Simule un jeu de données selon le modèle linéaire Gaussien simple.
    (Modèle (M.1))
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux.
        D doit être strictement supérieur à L.
        
    mu : 1-D ndarray,
        Vecteur de taille D, moyenne des observations.
    
    sigma2 : float,
        Réel positif, variance du bruit des observations.
    
    N : int,
        Nombre d'individus à simuler.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    Pour tout n entre 0 et N-1, si y et z sont les n-ièmes lignes respectives de Y et Z,
    alors y et z sont liés par la relation suivante : y = W.z + mu + epsilon[n],
    où epsilon[n] est un vecteur gaussien centré, indépendant de z, de variance sigma2*Id.
    '''
    D,L = np.shape(W)
    Z = rd.normal(0,1,(N,L))
    Y = np.array([W@z + mu + rd.normal(0,sigma2,D) for z in Z])
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
    return Z, Y

def param_cov(D,L,C,m1=0.0,m2=0.0,m3=0.0,m4=0.0,s1=1.0,s2=1.0,s3=1.0,s4=1.0,disp=False,orthog=True):
    '''Simule un ensemble de paramètres pour le modèle linéaire Gaussien mixte.
    (Modèle (M.2))
    
    Paramètres
    ----------
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D.
    
    C : int,
        Nombre de dimensions des vecteurs de covariables.
        Doit être inférieur ou égal à D.
    
    m1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 0.0 par défaut.
    
    m4 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 0.0 par défaut.
    
    s1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 1.0 par défaut.
    
    s4 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux et les effets fixes seront respectivement les colonnes d'une matrice de taille (D,L) et une matrice de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,s1²) et (m2,s2²).
        Si mis sur True, les axes principaux et les effets fixes seront respectivement les colonnes d'une matrice de taille (D,L) et une matrice de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,1) et (m2,1), passées par la fonction orthogonalize, puis multipliées par s1 et s2.
        Mis sur True par défaut.
        
    Renvois
    -------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux.
    
    V : 2-D ndarray,
        Matrice de taille (D,C) d'effets fixes.
        
    mu : 1-D ndarray,
        Vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3,s3²).
    
    sigma2 : float,
        Réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4,s4²).
    '''
    mu = rd.normal(m3,s3**2,D)
    sigma2 = rd.normal(m4,s4**2)**2
    
    if orthog:
        W = rd.normal(m1,1.0,(D,L))
        V = rd.normal(m2,1.0,(D,C))
        W = s1*mbu.orthogonalize(W)
        V = s2*mbu.orthogonalize(V)
    else :
        W = rd.normal(m1,s1**2,(D,L))
        V = rd.normal(m2,s2**2,(D,C))
    
    if disp:
        print('$W = $', W)
        print('$V = $', V)
        print('$\mu = $', mu)
        print('$\sigma^2 = $', sigma2)
        
    return W, V, mu, sigma2

def data_cov(W,V,mu,sigma2,N,Sigma_X=None,disp=False):
    '''Simule un jeu de données selon le modèle linéaire Gaussien mixte.
    (Modèle (M.2))
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice de taille (D,L) dont les colonnes sont les axes principaux.
        D doit être strictement supérieur à L.
    
    V : 2-D ndarray,
        Matrice de taille (D,C) d'effets fixes.
        D doit être supérieur ou égal à C.
    
    mu : 1-D ndarray,
        Vecteur de taille D, moyenne des observations.
    
    sigma2 : float,
        Réel positif, variance du bruit des observations.
    
    N : int,
        Nombre d'individus à simuler.
    
    Sigma_X : 2-D ndarray, optional,
        Matrice carrée d'ordre C, symétrique positive, de covariance des covariables.
        Si mis sur None, prend comme valeur la matrice identité d'ordre C.
        Mis sur None par défaut.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables simulés, indépendante de Z.
    
    Pour tout n entre 0 et N-1, si x, y et z sont les n-ièmes lignes respectives de X, Y et Z,
    alors x, y et z sont liés par la relation suivante : y = W.z + V.x + mu + epsilon[n],
    où epsilon[n] est un vecteur gaussien centré, indépendant de z et x, de variance sigma2*Id.
    '''
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

def param_LRPSI(D,L,m1=0.0,m2=0.0,s1=0.0,s2=0.0,la=1.0,p=0.1,disp=False,orthog=True):
    
    if orthog:
        W = rd.normal(m1,1.0,(D,L))
        W = s1*mbu.orthogonalize(W)
    else :
        W = rd.normal(m1,s1**2,(D,L))
    sigma2 = rd.normal(m2,s2**2)**2
    Lambda = sim_Lambda(D,la,p)
    
    return W, Lambda, sigma2

def data_LRPSI(W,Lambda,N,sigma2,disp=False):
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

def param_obsmix_1(K,D,L,m1=0.0,m2=0.0,m3=0.0,s1=1.0,s2=1.0,s3=1.0,disp=False,orthog=True):
    '''Simule K ensembles de paramètres pour la mixture sur l'espace observé de modèles linéaires Gaussiens simples.
    (Modèle (M.4.1))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D.
    
    m1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération des moyennes.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération des variances.
        Mis sur 0.0 par défaut.
    
    s1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération des moyennes.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération des variances.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux seront les colonnes de matrices de taille (D,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,s1²)
        Si mis sur True, les axes principaux seront les colonnes de matrices de taille (D,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,1.0), passée par la fonction orthogonalize, puis multipliée par s1.
        Mis sur True par défaut.
        
    Renvois
    -------
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux du cluster.
            - mu est un vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2,s2²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3,s3²).
    '''
    thetas = [param_lin(D,L,m1,m2,m3,s1,s2,s3,disp,orthog) for k in range(K)]
    return thetas

def param_obsmix_2(K,D,L,C,m1=0.0,m2=0.0,m3=0.0,m4=0.0,s1=1.0,s2=1.0,s3=1.0,s4=1.0,disp=False,orthog=True):
    '''Simule K ensembles de paramètres pour la mixture sur l'espace observé de modèles linéaires Gaussiens mixtes.
    (Modèle (M.4.2))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D.
    
    C : int,
        Nombre de dimensions des vecteurs de covariables.
        Doit être inférieur ou égal à D.
    
    m1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 0.0 par défaut.
    
    m4 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 0.0 par défaut.
    
    s1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 1.0 par défaut.
    
    s4 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux et les effets fixes seront respectivement les colonnes de matrices de taille (D,L) et des matrices de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,s1²) et (m2,s2²).
        Si mis sur True, les axes principaux et les effets fixes seront respectivement les colonnes de matrices de taille (D,L) et des matrices de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,1) et (m2,1), passées par la fonction orthogonalize, puis multipliées par s1 et s2.
        Mis sur True par défaut.
        
    Renvois
    -------
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux du cluster correspondant.
            - V est la matrice de taille (D,C) d'effets fixes du cluster correspondant.
            - mu est un vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3,s3²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4,s4²).
    '''
    thetas = [param_cov(D,L,C,m1,m2,m3,m4,s1,s2,s3,s4,disp,orthog) for k in range(K)]
    return thetas

def param_obsmix_3(K,D,L,m1=0.0,m2=0.0,s1=0.0,s2=0.0,la=1.0,p=0.1,disp=False,orthog=True):
    thetas = [param_LRPSI(D,L,m1,m2,s1,s2,la,p,disp,orthog) for k in range(K)]
    return thetas

def param_latmix_1(K,D,L,m2=0.0,m3=0.0,s2=1.0,s3=1.0,m_glob1=0.0,m_glob2=0.0,m_glob3=0.0,s_glob2=1.0,s_glob3=1.0,disp=False,orthog=True):
    '''Simule K ensembles de paramètres pour la mixture sur l'espace latent de modèles linéaires Gaussiens simples
    (Modèle (M.5.1)).
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D.
    
    m2 : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents.
        Mis sur 0.0 par défaut.
        
    s2 : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents.
        Mis sur 1.0 par défaut.
    
    m_glob1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m_glob2 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    m_glob3 : float, optional,
        Paramètre influent sur la génération de la variance du bruit des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    s_glob2 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs observés.
        Mis sur 1.0 par défaut.
    
    s_glob3 : float, optional,
        Paramètre influent sur la génération de la variance du bruit des vecteurs observés.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux seront les colonnes d'une matrice de taille (D,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob1,1).
        Si mis sur True, les axes principaux seront les colonnes d'une matrice de taille (D,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob1,1), passée par la fonction orthogonalize.
        Mis sur True par défaut.
        
    Renvois
    -------
    eta : list,
        Liste de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux.
            - mu est un vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob2,s_glob2²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m_glob3,s_glob3²).
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu,sigma2], où :
            - mu est un vecteur de taille L dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2,s2²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3,s3²).
    '''
    eta = param_lin(D,L,m_glob1,m_glob2,m_glob3,1.0,s_glob2,s_glob3,disp,orthog)
    thetas = [param_lin(L,L,0.0,m2,m3,1.0,s2,s3,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def param_latmix_2(K,D,L,C,m3=0.0,m4=0.0,s3=1.0,s4=1.0,m_glob1=0.0,m_glob2=0.0,m_glob3=0.0,m_glob4=0.0,s_glob2=1.0,s_glob3=1.0,s_glob4=1.0,disp=False,orthog=True):
    '''Simule K ensembles de paramètres pour la mixture sur l'espace latent de modèles linéaires Gaussiens mixtes, avec variables impertinentes.
    (Modèle (M.6.4))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D.
    
    C : int,
        Nombre de dimensions des vecteurs de covariables.
        Doit être inférieur ou égal à D.
    
    m3 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs latents.
        Mis sur 0.0 par défaut.
    
    m4 : float, optional,
        Paramètre influent sur la génération de la variance des vecteurs latents.
        Mis sur 0.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs latents.
        Mis sur 1.0 par défaut.
    
    s4 : float, optional,
        Paramètre influent sur la génération de la variance des vecteurs latents.
        Mis sur 1.0 par défaut.
    
    m_glob1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m_glob2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 0.0 par défaut.
    
    m_glob3 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    m_glob4 : float, optional,
        Paramètre influent sur la génération de la variance du bruit des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    s_glob2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 1.0 par défaut.
    
    s_glob3 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs observés.
        Mis sur 1.0 par défaut.
    
    s_glob4 : float, optional,
        Paramètre influent sur la génération de la variance du bruit des vecteurs observés.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux et les effets fixes seront respectivement les colonnes d'une matrice de taille (D,L) et une matrice de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob1,1) et (m_glob2,s_glob2²).
        Si mis sur True, les axes principaux et les effets fixes seront respectivement les colonnes d'une matrice de taille (D,L) et une matrice de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob1,1) et (m_glob2,1), passées par la fonction orthogonalize, puis multipliées par 1.0 et s_glob2.
        Mis sur True par défaut.
        
    Renvois
    -------
    eta : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,V,mu,sigma2], où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux.
            - V est une matrice de taille (D,C) d'effets fixes.
            - mu est un vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob3,s_glob3²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m_glob4,s_glob4²).
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu,sigma2], où :
            - mu est un vecteur de taille L dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3,s3²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4,s4²).
    '''
    eta = param_cov(D,L,C,m_glob1,m_glob2,m_glob3,m_glob4,1.0,s_glob2,s_glob3,s_glob4,disp,orthog)
    thetas = [param_lin(L,L,0.0,m3,m4,1.0,s3,s4,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def param_latmix_3(K,D,L,m2=0.0,m3=0.0,s2=1.0,s3=1.0,m_glob1=0.0,m_glob2=0.0,s_glob2=1.0,la=1.0,p=0.1,disp=False,orthog=True):
    eta = param_LRPSI(D,L,m_glob1,m_glob2,1.0,s_glob2,la,p,disp,orthog)
    thetas = [param_lin(L,L,0.0,m2,m3,1.0,s2,s3,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def sim_omega(N,K,N_min=2):
    '''Simule d'un vecteur de taille N rempli d'entiers compris entre 0 et K-1.
    
    Paramètres
    ----------
    N : int,
        Taille du vecteur à simuler.
    
    K : int,
        Nombre de clusters à simuler.
        Doit être strictement inférieur à N.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur renvoyé.
        Mis sur 2 par défaut.
        K*N_min doit être inférieur ou égal à N.
    
    Renvois
    -------
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    '''
    
    if N < K*N_min:
        K = int(N/N_min)
    
    nb_random = N - K*N_min
    random_part = rd.choice(K,nb_random)
    sorted_omega = np.concatenate([np.concatenate([k*np.ones(N_min) for k in range(K)]),random_part])
    omega = rd.permutation(sorted_omega)
    return omega.astype(int)

def data_obsmix_1(thetas,N,N_min=0,disp=False):
    '''Simule un jeu de données selon la mixture sur l'espace observé de modèles linéaires Gaussiens simples.
    (Modèle (M.4.1))
    
    Paramètres
    ----------
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux du cluster.
            - mu est un vecteur de taille D, supposé être la moyenne de chaque cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations de chaque cluster.
        D doit être strictement supérieur à L.
        
    N : int, optional,
        Nombre d'individus à simuler.
    
    N_min : int,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur L+1 par défaut.
        N doit être strictement supérieur à K*(L+1) et K*N_min.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    Pour tout n entre 0 et N-1, si y et z sont les n-ièmes lignes respectives de Y et Z,
    k est le n-ième coefficient de omega, et [W,mu,sigma2] est k-ième ensemble de paramètres de la liste thetas,
    alors y et z sont liés par la relation suivante : y = W.z + mu + epsilon[n],
    où epsilon[n] est un vecteur gaussien centré, indépendant de z, de variance sigma2*Id.
    '''
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

def data_obsmix_2(thetas,N,N_min=0,Sigma_X=None,disp=False):
    '''Simule un jeu de données selon la mixture sur l'espace observé de modèles linéaires Gaussiens mixtes.
    (Modèle (M.4.2)).
    
    Paramètres
    ----------
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux du cluster correspondant.
            - V est la matrice de taille (D,C) d'effets fixes du cluster correspondant.
            - mu est un vecteur de taille D, supposé être la moyenne de chaque cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations de chaque cluster.
        D doit être strictement supérieur à L et supérieur ou égal à C.
        
    N : int,
        Nombre d'individus à simuler.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur le max(L+1,C+1) par défaut.
        N doit être strictement supérieur à K*(L+1), K*(C+1) et K*N_min.
    
    Sigma_X : 2-D ndarray, optional,
        Matrice carrée d'ordre C, symétrique positive, de covariance des covariables.
        Si mis sur None, prend comme valeur la matrice identité d'ordre C.
        Mis sur None par défaut.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables simulés, indépendante de Z.
    
    Pour tout n entre 0 et N-1, si x, y et z sont les n-ièmes lignes respectives de X, Y et Z,
    k est le n-ième coefficient de omega, et [W,V,mu,sigma2] est k-ième ensemble de paramètres de la liste thetas,
    alors x, y et z sont liés par la relation suivante : y = W.z + V.x + mu + epsilon[n],
    où epsilon[n] est un vecteur gaussien centré, indépendant de z et x, de variance sigma2*Id.
    '''
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

def data_obsmix_3(thetas,N,N_min=0,disp=False):
    
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

def data_latmix_1(eta,thetas,N,N_min=2,disp=False):
    '''Simule un jeu de données selon le modèle linéaire Gaussien mixte sur l'espace latent.
    (Modèle (M.5.1))
    
    Paramètres
    ----------
    eta : list,
        Liste de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux.
            - mu est un vecteur de taille D supposé être la moyenne des observations.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations.
        D doit être strictement supérieur à L.
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu_k,sigma2_k], où :
            - mu_k est un vecteur de taille Lv, supposé être une moyenne locale de vecteurs latents dont la loi change selon le cluster.
            - sigma2_k est un réel positif, supposé être une variance locale de vecteurs latents dont la loi change selon le cluster.
           
    N : int,
        Nombre d'individus à simuler.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur 2 par défaut.
        N doit être strictement supérieur à K*N_min.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    Pour tout n entre 0 et N-1, si y et z sont les n-ièmes lignes respectives de Y et Z,
    alors y et z sont liés par la relation suivante : y = W.z + mu + epsilon[n],
    où epsilon[n] est un vecteur gaussien centré, indépendant de z, de variance sigma2*Id.
    '''
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

def data_latmix_2(eta,thetas,N,N_min=2,Sigma_X=None,disp=False):
    '''Simule un jeu de données selon le modèle linéaire Gaussien mixte sur l'espace latent, avec covariables.
    (Modèle (M.5.2))
    
    Paramètres
    ----------
    eta : list,
        Liste de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (D,L) dont les colonnes sont les axes principaux.
            - V est une matrice de taille (D,C) d'effets fixes.
            - mu est un vecteur de taille D supposé être la moyenne des observations.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations.
        D doit être strictement supérieur à L et supérieur ou égal à C.
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu_k,sigma2_k], où :
            - mu_k est un vecteur de taille Lv, supposé être une moyenne locale de vecteurs latents dont la loi change selon le cluster.
            - sigma2_k est un réel positif, supposé être une variance locale de vecteurs latents dont la loi change selon le cluster.
           
    N : int,
        Nombre d'individus à simuler.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur 2 par défaut.
        N doit être strictement supérieur à K*N_min.
    
    Sigma_X : 2-D ndarray, optional,
        Matrice carrée d'ordre C, symétrique positive, de covariance des covariables.
        Si mis sur None, prend comme valeur la matrice identité d'ordre C.
        Mis sur None par défaut.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    Pour tout n entre 0 et N-1, si x, y et z sont les n-ièmes lignes respectives de X, Y et Z,
    alors x, y et z sont liés par la relation suivante : y = W.z + V.x + mu + epsilon[n],
    où epsilon[n] est un vecteur gaussien centré, indépendant de z et x, de variance sigma2*Id.
    '''
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

def data_latmix_3(eta,thetas,N,N_min=2,Sigma_X=None,disp=False):
    
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
        
    X = np.array([rd.multivariate_normal(np.zeros(D),nla.inv(Lambda)) for n in range(N)])
    noise = rd.normal(0,sigma2,(N,D))
    
    Y = Z@np.transpose(W) + X + mu + noise
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
    
    return Z, omega, Y, X

def noisy_param_1(D,L,U,m1=0.0,m2=0.0,m3=0.0,s1=1.0,s2=1.0,s3=1.0,disp=False,orthog=True):
    '''Simule un ensemble de paramètres pour le modèle linéaire Gaussien simple avec variables impertinentes.
    (Modèle (M.6.1))
    
    Paramètres
    ----------
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D-1.
    
    U : int,
        Nombre de variables pertinentes de l'espace observé.
        Doit être strictement comprsi entre L et D.
    
    m1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 0.0 par défaut.
    
    s1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux seront les colonnes d'une matrice de taille (U,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,s1²)
        Si mis sur True, les axes principaux seront les colonnes d'une matrice de taille (U,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,1.0), passée par la fonction orthogonalize, puis multipliée par s1.
        Mis sur True par défaut.
        
    Renvois
    -------
    W : 2-D ndarray,
        Matrice de taille (U,L) dont les colonnes sont les axes principaux.
        
    mu : 1-D ndarray,
        Vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2,s2²).
    
    sigma2 : float,
        Réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3,s3²).
    '''
    mu = rd.normal(m2,s2**2,D)
    sigma2 = rd.normal(m3,s3**2)**2
    
    if orthog:
        W = rd.normal(m1,1.0,(U,L))
        W = s1*mbu.orthogonalize(W)
    else:
        W = rd.normal(m1,s1**2,(U,L))
    
    if disp:
        print('$W = $', W)
        print('$\mu = $', mu)
        print('$\sigma^2 = $', sigma2)
        
    return W, mu, sigma2

def noisy_data_1(W,mu,sigma2,N,disp=False):
    '''Simule un jeu de données selon le modèle linéaire Gaussien avec variables impertinentes.
    (Modèle (M.6.1))
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice de taille (U,L) dont les colonnes sont les axes principaux.
        U doit être strictement supérieur à L.
        
    mu : 1-D ndarray,
        Vecteur de taille D, moyenne des observations.
        D doit être strictement supérieur à U.
    
    sigma2 : float,
        Réel positif, variance du bruit des observations.
    
    N : int,
        Nombre d'individus à simuler.
    
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    
    Pour tout n entre 0 et N-1, si y et z sont les n-ièmes lignes respectives de Y et Z,
    et R est la matrice ra_matrix(iota), alors y et z sont liés par la relation suivante :
    y = R.Concatenate(W.z,0_{D-U}) + mu + epsilon[n], où 0_{D-U} est le vecteur nul de taille D-U,
    et epsilon[n] est un vecteur gaussien centré, indépendant de z, de variance sigma2*Id.
    '''
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

def noisy_param_2(D,L,C,U,m1=0.0,m2=0.0,m3=0.0,m4=0.0,s1=1.0,s2=1.0,s3=1.0,s4=1.0,disp=False,orthog=True):
     '''Simule un ensemble de paramètres pour le modèle linéaire Gaussien mixte avec variables impertinentes.
     (Modèle (M.6.2))
    
    Paramètres
    ----------
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D-1.
    
    C : int,
        Nombre de dimensions des vecteurs de covariables.
        Doit être inférieur ou égal à D.
    
    U : int,
        Nombre de variables pertinentes de l'espace observé.
        Doit être strictement compris entre L et D.
    
    m1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 0.0 par défaut.
    
    m4 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 0.0 par défaut.
    
    s1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 1.0 par défaut.
    
    s4 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux et les effets fixes seront respectivement les colonnes d'une matrice de taille (U,L) et une matrice de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,s1²) et (m2,s2²).
        Si mis sur True, les axes principaux et les effets fixes seront respectivement les colonnes d'une matrice de taille (U,L) et une matrice de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,1) et (m2,1), passées par la fonction orthogonalize, puis multipliées par s1 et s2.
        Mis sur True par défaut.
        
    Renvois
    -------
    W : 2-D ndarray,
        Matrice de taille (U,L) dont les colonnes sont les axes principaux.
    
    V : 2-D ndarray,
        Matrice de taille (D,C) d'effets fixes.
        
    mu : 1-D ndarray,
        Vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3,s3²).
    
    sigma2 : float,
        Réel positif obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4,s4²).
    '''
    mu = rd.normal(m3,s3**2,D)
    sigma2 = rd.normal(m4,s4**2)**2
    
    if orthog:
        W = rd.normal(m1,1.0,(U,L))
        V = rd.normal(m2,1.0,(D,C))
        W = s1*mbu.orthogonalize(W)
        V = s2*mbu.orthogonalize(V)
    else:
        W = rd.normal(m1,s1**2,(U,L))
        V = rd.normal(m2,s2**2,(D,C))
    
    if disp:
        print('$W = $', W)
        print('$V = $', V)
        print('$\mu = $', mu)
        print('$\sigma^2 = $', sigma2)
        
    return W, V, mu, sigma2

def noisy_data_2(W,V,mu,sigma2,N,Sigma_X=None,disp=False):
    '''Simule un jeu de données selon le modèle linéaire Gaussien avec covariables et variables impertinentes.
    (Modèle (M.6.2))
    
    Paramètres
    ----------
    W : 2-D ndarray,
        Matrice de taille (U,L) dont les colonnes sont les axes principaux.
        U doit être supérieur ou égal à L.
    
    V : 2-D ndarray,
        Matrice de taille (D,C) d'effets fixes.
        D doit être supérieur ou égal à C, et strictement supérieur à U.
    
    mu : 1-D ndarray,
        Vecteur de taille D, moyenne des observations.
    
    sigma2 : float,
        Réel positif, variance du bruit des observations.
    
    N : int,
        Nombre d'individus à simuler.
    
    Sigma_X : 2-D ndarray, optional,
        Matrice carrée d'ordre C, symétrique positive, de covariance des covariables.
        Si mis sur None, prend comme valeur la matrice identité d'ordre C.
        Mis sur None par défaut.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables simulés, indépendante de Z.
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    
    Pour tout n entre 0 et N-1, si x, y et z sont les n-ièmes lignes respectives de X, Y et Z,
    et R est la matrice ra_matrix(iota), alors x, y et z sont liés par la relation suivante :
    y = R.Concatenate(W.z,0_{D-U}) + V.x + mu + epsilon[n], où 0_{D-U} est le vecteur nul de taille D-U,
    et epsilon[n] est un vecteur gaussien centré, indépendant de z, de variance sigma2*Id.
    '''
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

def noisy_param_obsmix_1(K,D,L,U,m1=0.0,m2=0.0,m3=0.0,s1=1.0,s2=1.0,s3=1.0,disp=False,orthog=True):
    '''Simule K ensembles de paramètres pour le modèle linéaire Gaussien mixte sur l'espace des observations, avec variables impertinentes.
    (Modèle (M.6.5))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D-1.
    
    U : int,
        Nombre de variables pertinentes de l'espace observé.
        Doit être strictement compris entre L et D.
    
    m1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération des moyennes.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération des variances.
        Mis sur 0.0 par défaut.
    
    s1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération des moyennes.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération des variances.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux seront les colonnes de matrices de taille (U,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,s1²)
        Si mis sur True, les axes principaux seront les colonnes de matrices de taille (U,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,1.0), passée par la fonction orthogonalize, puis multipliée par s1.
        Mis sur True par défaut.
        
    Renvois
    -------
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux du cluster.
            - mu est un vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2,s2²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3,s3²).
    '''
    thetas = [noisy_param_1(D,L,U,m1,m2,m3,s1,s2,s3,disp,orthog) for k in range(K)]
    return thetas

def noisy_param_obsmix_2(K,D,L,C,U,m1=0.0,m2=0.0,m3=0.0,m4=0.0,s1=1.0,s2=1.0,s3=1.0,s4=1.0,disp=False,orthog=True):
    '''Simule un ensemble de paramètres pour le modèle linéaire Gaussien mixte sur l'espace observé, avec covariables et variables impertinentes.
    (Modèle (M.6.6))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D-1.
    
    C : int,
        Nombre de dimensions des vecteurs de covariables.
        Doit être inférieur ou égal à D.
    
    U : int,
        Nombre de dimensions pertinentes de l'espace observé.
        Doit être strictement compris entre L et D.
    
    m1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 0.0 par défaut.
    
    m4 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 0.0 par défaut.
    
    s1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération de la moyenne.
        Mis sur 1.0 par défaut.
    
    s4 : float, optional,
        Paramètre influent sur la génération de la variance.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux et les effets fixes seront respectivement les colonnes de matrices de taille (U,L) et des matrices de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,s1²) et (m2,s2²).
        Si mis sur True, les axes principaux et les effets fixes seront respectivement les colonnes de matrices de taille (U,L) et des matrices de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1,1) et (m2,1), passées par la fonction orthogonalize, puis multipliées par s1 et s2.
        Mis sur True par défaut.
        
    Renvois
    -------
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,V,mu,sigma2], où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux du cluster correspondant.
            - V est la matrice de taille (D,C) d'effets fixes du cluster correspondant.
            - mu est un vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3,s3²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4,s4²).
    '''
    thetas = [noisy_param_2(D,L,C,U,m1,m2,m3,m4,s1,s2,s3,s4,disp,orthog) for k in range(K)]
    return thetas

def noisy_param_latmix_1(K,D,L,U,m2=0.0,m3=0.0,s2=1.0,s3=1.0,m_glob1=0.0,m_glob2=0.0,m_glob3=0.0,s_glob2=1.0,s_glob3=1.0,disp=False,orthog=True):
    '''Simule K ensembles de paramètres pour la mixture sur l'espace latent de modèles linéaires Gaussiens simples, avec variables impertinentes.
    (Modèle (M.6.3))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D-1.
    
    U : int,
        Nombre de dimensions pertinentes de l'espace observé.
        Doit être strictement compris entre L et D.
    
    m2 : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents.
        Mis sur 0.0 par défaut.
    
    m3 : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents.
        Mis sur 0.0 par défaut.
        
    s2 : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents.
        Mis sur 1.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents.
        Mis sur 1.0 par défaut.
    
    m_glob1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m_glob2 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    m_glob3 : float, optional,
        Paramètre influent sur la génération de la variance du bruit des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    s_glob2 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs observés.
        Mis sur 1.0 par défaut.
    
    s_glob3 : float, optional,
        Paramètre influent sur la génération de la variance du bruit des vecteurs observés.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux seront les colonnes d'une matrice de taille (U,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob1,1).
        Si mis sur True, les axes principaux seront les colonnes d'une matrice de taille (U,L) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob1,1), passée par la fonction orthogonalize.
        Mis sur True par défaut.
        
    Renvois
    -------
    eta : list,
        Liste de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux.
            - mu est un vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob2,s_glob2²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m_glob3,s_glob3²).
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu,sigma2], où :
            - mu est un vecteur de taille L dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2,s2²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3,s3²).
    '''
    eta = noisy_param_1(D,L,U,m_glob1,m_glob2,m_glob3,1.0,s_glob2,s_glob3,disp,orthog)
    thetas = [param_lin(L,L,0.0,m2,m3,1.0,s2,s3,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def noisy_param_latmix_2(K,D,L,C,U,m3=0.0,m4=0.0,s3=1.0,s4=1.0,m_glob1=0.0,m_glob2=0.0,m_glob3=0.0,m_glob4=0.0,s_glob2=1.0,s_glob3=1.0,s_glob4=1.0,disp=False,orthog=True):
    '''Simule un ensemble de paramètres pour le modèle linéaire Gaussien mixte sur l'espace latent, avec covariables et variables et variables impertinentes.
    (Modèle (M.6.4))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    D : int,
        Nombre de dimensions de l'espace observé.
    
    L : int,
        Nombre de dimensions de l'espace latent.
        Doit être strictement inférieur à D-1.
    
    C : int,
        Nombre de dimensions des vecteurs de covariables.
        Doit être inférieur ou égal à D.
    
    U : int,
        Nombre de dimensions pertinentes de l'espace observé.
        Doit être strictement compris entre L et D.
    
    m3 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs latents.
        Mis sur 0.0 par défaut.
    
    m4 : float, optional,
        Paramètre influent sur la génération de la variance des vecteurs latents.
        Mis sur 0.0 par défaut.
    
    s3 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs latents.
        Mis sur 1.0 par défaut.
    
    s4 : float, optional,
        Paramètre influent sur la génération de la variance des vecteurs latents.
        Mis sur 1.0 par défaut.
    
    m_glob1 : float, optional,
        Paramètre influent sur la génération des axes principaux.
        Mis sur 0.0 par défaut.
    
    m_glob2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 0.0 par défaut.
    
    m_glob3 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    m_glob4 : float, optional,
        Paramètre influent sur la génération de la variance du bruit des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    s_glob2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 1.0 par défaut.
    
    s_glob3 : float, optional,
        Paramètre influent sur la génération de la moyenne des vecteurs observés.
        Mis sur 1.0 par défaut.
    
    s_glob4 : float, optional,
        Paramètre influent sur la génération de la variance du bruit des vecteurs observés.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux et les effets fixes seront respectivement les colonnes d'une matrice de taille (U,L) et une matrice de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob1,1) et (m_glob2,s_glob2²).
        Si mis sur True, les axes principaux et les effets fixes seront respectivement les colonnes d'une matrice de taille (U,L) et une matrice de taille (D,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob1,1) et (m_glob2,1), passées par la fonction orthogonalize, puis multipliées par 1.0 et s_glob2.
        Mis sur True par défaut.
        
    Renvois
    -------
    eta : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,V,mu,sigma2], où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux.
            - V est une matrice de taille (D,C) d'effets fixes.
            - mu est un vecteur de taille D dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m_glob3,s_glob3²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m_glob4,s_glob4²).
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu,sigma2], où :
            - mu est un vecteur de taille L dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3,s3²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4,s4²).
    '''
    eta = noisy_param_2(D,L,C,U,m_glob1,m_glob2,m_glob3,m_glob4,1.0,s_glob2,s_glob3,s_glob4,disp,orthog)
    thetas = [param_lin(L,L,0.0,m3,m4,1.0,s3,s4,disp,orthog)[1:] for k in range(K)]
    return eta, thetas

def noisy_data_obsmix_1(thetas,N,N_min=0,disp=False):
    '''Simule un jeu de données selon la mixture sur l'espace observé de modèles linéaires Gaussiens simples, avec variables impertinentes.
    (Modèle (M.6.5)).
    
    Paramètres
    ----------
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux du cluster.
            - mu est un vecteur de taille D, supposé être la moyenne de chaque cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations de chaque cluster.
        On doit avoir D > U > L.
        
    N : int, optional,
        Nombre d'individus à simuler.
    
    N_min : int,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur L+1 par défaut.
        N doit être strictement supérieur à K*(L+1) et K*N_min.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    iotas : 2-D ndarray,
        Matrice de taille (K,D) dont chaque ligne contient U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    
    Pour tout n entre 0 et N-1, si y et z sont les n-ièmes lignes respectives de Y et Z,
    k est le n-ième coefficient de omega, [W,mu,sigma2] est k-ième ensemble de paramètres de la liste thetas,
    iota est la k-ième ligne de iotas et R est la matrice ra_matrix(iota), alors y et z sont liés par la
    relation suivante : y = R.Concatenate(W.z,0_{D-U}) + mu + epsilon[n], où 0_{D-U} est le vecteur nul de
    taille D-U, et epsilon[n] est un vecteur gaussien centré, indépendant de z, de variance sigma2*Id.
    '''
    K = len(thetas)
    U,L = np.shape(thetas[0][0])
    D = len(thetas[0][1])
    
    omega = sim_omega(N,K,int(max(L+1,N_min)))
    K = int(max(omega)) + 1
    
    Z = rd.normal(0,1,(N,L))
    iotas_prov = np.concatenate([np.ones((K,U)),np.zeros((K,D-U))],axis=1).astype(int)
    iotas = np.array([rd.permutation(iota) for iota in iotas_prov])
    
    Y = drr.FS_mixrec1(thetas,Z,omega,iotas) + np.array([rd.normal(0,thetas[omega[n]][2],D) for n in range(N)])
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\omega = $', omega)
        print('$\iotas = $', iotas)
        
    return Z, omega, Y, iotas

def noisy_data_obsmix_2(thetas,N,N_min=0,Sigma_X=None,disp=False):
    '''Simule un jeu de données selon la mixture sur l'espace observé de modèles linéaires Gaussiens mixtes, avec variables impertinentes.
    (Modèle (M.6.6)).
    
    Paramètres
    ----------
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux du cluster correspondant.
            - V est la matrice de taille (D,C) d'effets fixes du cluster correspondant.
            - mu est un vecteur de taille D, supposé être la moyenne de chaque cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations de chaque cluster.
        On doit avoir D > U > L, et C doit être inférieur ou égal à D.
        
    N : int,
        Nombre d'individus à simuler.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur max(L+1,C+1) par défaut.
        N doit être strictement supérieur à K*(L+1), K*(C+1) et K*N_min.
    
    Sigma_X : 2-D ndarray, optional,
        Matrice carrée d'ordre C, symétrique positive, de covariance des covariables.
        Si mis sur None, prend comme valeur la matrice identité d'ordre C.
        Mis sur None par défaut.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables simulés, indépendante de Z.
    
    iotas : 2-D ndarray,
        Matrice de taille (K,D) dont chaque ligne contient U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    
    Pour tout n entre 0 et N-1, si x, y et z sont les n-ièmes lignes respectives de X, Y et Z,
    k est le n-ième coefficient de omega, [W,V,mu,sigma2] est k-ième ensemble de paramètres de la liste thetas,
    iota est la k-ième ligne de iotas et R est la matrice ra_matrix(iota), alors x, y et z sont liés par
    la relation suivante : y = R.Concatenate(W.z,0_{D-U}) + V.x + mu + epsilon[n], où 0_{D-U} est le vecteur nul
    de taille D-U, et epsilon[n] est un vecteur gaussien centré, indépendant de z, de variance sigma2*Id.
    '''
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
    
    Y = drr.FS_mixrec2(thetas,Z,X,omega,iotas) + np.array([rd.normal(0,thetas[omega[n]][3],D) for n in range(N)])
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
        print('$\iotas = $', iotas)
        
    return Z, omega, Y, X, iotas

def noisy_data_latmix_1(eta,thetas,N,N_min=2,disp=False):
    '''Simule un jeu de données selon le modèle linéaire Gaussien mixte sur l'espace latent, avec variables impertinentes.
    (Modèle (M.6.3))
    
    Paramètres
    ----------
    eta : list,
        Liste de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux.
            - mu est un vecteur de taille D supposé être la moyenne des observations.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations.
        On doit avoir : D > U > L.
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu_k,sigma2_k], où :
            - mu_k est un vecteur de taille Lv, supposé être une moyenne locale de vecteurs latents dont la loi change selon le cluster.
            - sigma2_k est un réel positif, supposé être une variance locale de vecteurs latents dont la loi change selon le cluster.
           
    N : int,
        Nombre d'individus à simuler.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur 2 par défaut.
        N doit être strictement supérieur à K*N_min.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
    
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    
    Pour tout n entre 0 et N-1, si y et z sont les n-ièmes lignes respectives de Y et Z,
    et R est la matrice ra_matrix(iota), alors y et z sont liés par la relation suivante :
    y = R.Concatenate(W.z,0_{D-U}) + mu + epsilon[n], où 0_{D-U} est le vecteur nul de taille D-U,
    et epsilon[n] est un vecteur gaussien centré, indépendant de z, de variance sigma2*Id.
    '''
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
    
    Y = drr.restit(Y_prov,iota) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\omega = $', omega)
        print('$\iota = $', iota)
    
    return Z, omega, Y, iota

def noisy_data_latmix_2(eta,thetas,N,N_min=2,Sigma_X=None,disp=False):
    '''Simule un jeu de données selon le modèle linéaire Gaussien mixte sur l'espace latent, avec covariables.
    (Modèle (M.6.4))
    
    Paramètres
    ----------
    eta : list,
        Liste de paramètres de la forme [W,mu,sigma2], où :
            - W est une matrice de taille (U,L) dont les colonnes sont les axes principaux.
            - V est une matrice de taille (D,C) d'effets fixes.
            - mu est un vecteur de taille D supposé être la moyenne des observations.
            - sigma2 est un réel positif, supposé être la variance du bruit des observations.
        On doit avoir : D > U > L, et C doit être inférieur ou égal à D.
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu_k,sigma2_k], où :
            - mu_k est un vecteur de taille Lv, supposé être une moyenne locale de vecteurs latents dont la loi change selon le cluster.
            - sigma2_k est un réel positif, supposé être une variance locale de vecteurs latents dont la loi change selon le cluster.
           
    N : int,
        Nombre d'individus à simuler.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur 2 par défaut.
        N doit être strictement supérieur à K*N_min.
    
    Sigma_X : 2-D ndarray, optional,
        Matrice carrée d'ordre C, symétrique positive, de covariance des covariables.
        Si mis sur None, prend comme valeur la matrice identité d'ordre C.
        Mis sur None par défaut.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Z : 2-D ndarray,
        Matrice de taille (N,L) dont les lignes sont les vecteurs latents simulés.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,D) dont les lignes sont les vecteurs observés obtenus.
        
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables simulés, indépendante de Z.
        
    iota : 1-D ndarray,
        Vecteur de taille D contenant U fois le nombre 1 et D-U fois le nombre 0, où le nombre 1 signifie que la dimension correspondante est pertinente, et le nombre 0 signifie qu'elle ne l'est pas.
    
    Pour tout n entre 0 et N-1, si x, y et z sont les n-ièmes lignes respectives de X, Y et Z,
    et R est la matrice ra_matrix(iota), alors x, y et z sont liés par la relation suivante :
    y = R.Concatenate(W.z,0_{D-U}) + V.x + mu + epsilon[n], où 0_{D-U} est le vecteur nul de taille D-U,
    et epsilon[n] est un vecteur gaussien centré, indépendant de z et x, de variance sigma2*Id.
    '''
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
    
    Y = drr.restit(Y_prov,iota) + X@np.transpose(V) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
        print('$\iota = $', iota)
    
    return Z, omega, Y, X, iota

def noisy_param_spemix_1(K,Dv,Du,Lv,Lu,m1u=0.0,m1v=0.0,m2u=0.0,m2v=0.0,m2t=0.0,m3u=0.0,m3v=0.0,m3t=0.0,s1u=1.0,s1v=1.0,s2u=1.0,s2v=1.0,s2t=1.0,s3u=1.0,s3v=1.0,s3t=1.0,disp=False,orthog=True):
    '''Simule K ensembles de paramètres pour la mixture sur seulement certaines dimensions des espaces observé et latent de modèles linéaires Gaussiens simples.
    (Modèle (M.6.7))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    Dv : int,
        Nombre de dimensions de l'espace observé sur lesquelles le modèle est mixte.
    
    Du : int,
        Nombre de dimensions de l'espace observé sur lesquelles le modèle n'est pas mixte.
    
    Lv : int,
        Nombre de dimensions de l'espace latent sur lesquelles le modèle est mixte.
        Doit être strictement inférieur à Dv.
    
    Lu : int,
        Nombre de dimensions de l'espace latent sur lesquelles le modèle n'est pas mixte.
        Doit être strictement inférieur à Du.
    
    m1u : float, optional,
        Paramètre influent sur la génération des axes principaux des dimensions "non-mixtes".
        Mis sur 0.0 par défaut.
        
    m1v : float, optional,
        Paramètre influent sur la génération des axes principaux des dimensions "mixtes".
        Mis sur 0.0 par défaut.
    
    m2u : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents dont la loi ne dépend pas du cluster.
        Mis sur 0.0 par défaut.
        
    m2v : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents dont la loi dépend du cluster.
        Mis sur 0.0 par défaut.
    
    m2t : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    m3u : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents dont la loi ne dépend pas du cluster.
        Mis sur 0.0 par défaut.
        
    m3v : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents dont la loi dépend du cluster.
        Mis sur 0.0 par défaut.
    
    m3t : float, optional,
        Paramètre influent sur la génération des variances des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    s1u : float, optional,
        Paramètre influent sur la génération des axes principaux des dimensions "non-mixtes".
        Mis sur 1.0 par défaut.
        
    s1v : float, optional,
        Paramètre influent sur la génération des axes principaux des dimensions "mixtes".
        Mis sur 1.0 par défaut.
    
    s2u : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents dont la loi ne dépend pas du cluster.
        Mis sur 1.0 par défaut.
        
    s2v : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents dont la loi dépend du cluster.
        Mis sur 1.0 par défaut.
    
    s2t : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs observés.
        Mis sur 1.0 par défaut.
    
    s3u : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents dont la loi ne dépend pas du cluster.
        Mis sur 1.0 par défaut.
        
    s3v : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents dont la loi dépend du cluster.
        Mis sur 1.0 par défaut.
    
    s3t : float, optional,
        Paramètre influent sur la génération des variances des vecteurs observés.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux des dimensions "mixtes" et "non-mixtes" seront respectivement les colonnes d'une matrice de taille (Dv,Lv) et (Du,Lu) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1v,s1v²) et (m1u,s1u²).
        Si mis sur True, les axes principaux des dimensions "mixtes" et "non-mixtes" seront respectivement les colonnes d'une matrice de taille (Dv,Lv) et (Du,Lu) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1v,1) et (m1u,s1), passée par la fonction orthogonalize puis multipliée par s1v et s1u.
        Mis sur True par défaut.
        
    Renvois
    -------
    eta : list,
        Liste de paramètres de la forme [Wv,Wu,mu,nu,sigma2,tau2], où :
            - Wv est une matrice de taille (Dv,Lv) dont les colonnes sont les axes principaux des dimensions "mixtes".
            - Wu est une matrice de taille (Du,Lu) dont les colonnes sont les axes principaux des dimensions "non-mixtes".
            - mu est un vecteur de taille Dv+Du dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2t,s2t²).
            - nu est un vecteur de taille Lu dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2u,s2u²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3t,s3t²).
            - tau2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3u,s3u²).
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu,sigma2], où :
            - mu est un vecteur de taille Lv dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m2v,s2v²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m3v,s3v²).
    '''
    D = Du+Dv
    mu = rd.normal(m2t,s2t**2,D)
    nu = rd.normal(m2u,s2u**2,Lu)
    sigma2 = rd.normal(m3t,s3t**2)**2
    tau2 = rd.normal(m3u,s3u**2)**2
    
    thetas = [[] for k in range(K)]    
    for k in range(K):
        mu_k = rd.normal(m2v,s2v**2,Lv)
        sigma2_k = rd.normal(m3v,s3v**2)**2
        thetas[k] = [mu_k,sigma2_k]    
    
    if orthog:
        Wu = rd.normal(m1u,1.0,(Du,Lu))
        Wv = rd.normal(m1v,1.0,(Dv,Lv))
        Wu = s1u*mbu.orthogonalize(Wu)
        Wv = s1v*mbu.orthogonalize(Wv)
    else:
        Wu = rd.normal(m1u,s1u**2,(Du,Lu))
        Wv = rd.normal(m1v,s1v**2,(Dv,Lv))
    
    eta = [Wv,Wu,mu,nu,sigma2,tau2]
    
    if disp:
        print("eta =",eta)
        print("thetas =",thetas)
        
    return eta, thetas

def noisy_param_spemix_2(K,Dv,Du,Lv,Lu,C,m1u=0.0,m1v=0.0,m2=0.0,m3u=0.0,m3v=0.0,m3t=0.0,m4u=0.0,m4v=0.0,m4t=0.0,s1u=1.0,s1v=1.0,s2=1.0,s3u=1.0,s3v=1.0,s3t=1.0,s4u=1.0,s4v=1.0,s4t=1.0,disp=False,orthog=True):
    '''Simule K ensembles de paramètres pour la mixture sur seulement certaines dimensions des espaces observé et latent de modèles linéaires Gaussiens mixtes.
    (Modèle (M.6.8))
    
    Paramètres
    ----------
    K : int,
        Nombre de clusters.
    
    Dv : int,
        Nombre de dimensions de l'espace observé sur lesquelles le modèle est mixte.
    
    Du : int,
        Nombre de dimensions de l'espace observé sur lesquelles le modèle n'est pas mixte.
    
    Lv : int,
        Nombre de dimensions de l'espace latent sur lesquelles le modèle est mixte.
        Doit être strictement inférieur à Dv.
    
    Lu : int,
        Nombre de dimensions de l'espace latent sur lesquelles le modèle n'est pas mixte.
        Doit être strictement inférieur à Du.
    
    C : int,
        Nombre de dimensions des vecteurs de covariables.
        Doit être inférieur ou égal à Dv+Du.
    
    m1u : float, optional,
        Paramètre influent sur la génération des axes principaux des dimensions "non-mixtes".
        Mis sur 0.0 par défaut.
        
    m1v : float, optional,
        Paramètre influent sur la génération des axes principaux des dimensions "mixtes".
        Mis sur 0.0 par défaut.
    
    m2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 0.0 par défaut.
    
    m3u : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents dont la loi ne dépend pas du cluster.
        Mis sur 0.0 par défaut.
        
    m3v : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents dont la loi dépend du cluster.
        Mis sur 0.0 par défaut.
    
    m3t : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    m4u : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents dont la loi ne dépend pas du cluster.
        Mis sur 0.0 par défaut.
        
    m4v : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents dont la loi dépend du cluster.
        Mis sur 0.0 par défaut.
    
    m4t : float, optional,
        Paramètre influent sur la génération des variances des vecteurs observés.
        Mis sur 0.0 par défaut.
    
    s1u : float, optional,
        Paramètre influent sur la génération des axes principaux des dimensions "non-mixtes".
        Mis sur 1.0 par défaut.
        
    s1v : float, optional,
        Paramètre influent sur la génération des axes principaux des dimensions "mixtes".
        Mis sur 1.0 par défaut.
    
    s2 : float, optional,
        Paramètre influent sur la génération des effets fixes.
        Mis sur 1.0 par défaut.
    
    s3u : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents dont la loi ne dépend pas du cluster.
        Mis sur 1.0 par défaut.
        
    s3v : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs latents dont la loi dépend du cluster.
        Mis sur 1.0 par défaut.
    
    s3t : float, optional,
        Paramètre influent sur la génération des moyennes des vecteurs observés.
        Mis sur 1.0 par défaut.
    
    s4u : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents dont la loi ne dépend pas du cluster.
        Mis sur 1.0 par défaut.
        
    s4v : float, optional,
        Paramètre influent sur la génération des variances des vecteurs latents dont la loi dépend du cluster.
        Mis sur 1.0 par défaut.
    
    s4t : float, optional,
        Paramètre influent sur la génération des variances des vecteurs observés.
        Mis sur 1.0 par défaut.
        
    disp : bool, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les paramètres obtenus.
        Mis sur False par défaut.
    
    orthog : bool, optional,
        Si mis sur False, les axes principaux des dimensions "mixtes" et "non-mixtes" et les effets fixes seront respectivement les colonnes d'une matrice de taille (Dv,Lv), (Du,Lu) et (Dv+Du,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1v,s1v²), (m1u,s1u²) et (m2,s2²).
        Si mis sur True, les axes principaux des dimensions "mixtes" et "non-mixtes" et les effets fixes seront respectivement les colonnes d'une matrice de taille (Dv,Lv), (Du,Lu) et (Dv+Du,C) dont les coefficients seront des variables aléatoires gaussiennes i.i.d. de paramètres (m1v,1), (m1u,s1) et (m2,1) passée par la fonction orthogonalize puis multipliée par s1v, s1u et s2.
        Mis sur True par défaut.
        
    Renvois
    -------
    eta : list,
        Liste de paramètres de la forme [Wv,Wu,V,mu,nu,sigma2,tau2], où :
            - Wv est une matrice de taille (Dv,Lv) dont les colonnes sont les axes principaux des dimensions "mixtes".
            - Wu est une matrice de taille (Du,Lu) dont les colonnes sont les axes principaux des dimensions "non-mixtes".
            - V est une matrice de taille (D,C) d'effets fixes.
            - mu est un vecteur de taille Dv+Du dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3t,s3t²).
            - nu est un vecteur de taille Lu dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3u,s3u²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4t,s4t²).
            - tau2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4u,s4u²).
    
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu,sigma2], où :
            - mu est un vecteur de taille Lv dont les coefficients sont des variables aléatoires gaussiennes i.i.d. de paramètres (m3v,s3v²).
            - sigma2 est un réel positif, obtenu en élevant au carré une variable aléatoire gaussienne de paramètres (m4v,s4v²).
    '''
    D = Du+Dv
    
    mu = rd.normal(m3t,s3t**2,D)
    nu = rd.normal(m3u,s3u**2,Lu)
    sigma2 = rd.normal(m4t,s4t**2)**2
    tau2 = rd.normal(m4u,s4u**2)**2
    
    thetas = [[] for k in range(K)]    
    for k in range(K):
        mu_k = rd.normal(m3v,s3v**2,Lv)
        sigma2_k = rd.normal(m4v,s4v**2)**2
        thetas[k] = [mu_k,sigma2_k]    
    
    if orthog:
        Wu = rd.normal(m1u,1.0,(Du,Lu))
        Wv = rd.normal(m1v,1.0,(Dv,Lv))
        V = rd.normal(m2,1.0,(D,C))
        Wu = s1u*mbu.orthogonalize(Wu)
        Wv = s1v*mbu.orthogonalize(Wv)
        V = s2*mbu.orthogonalize(V)
    else:
        Wu = rd.normal(m1u,s1u**2,(Du,Lu))
        Wv = rd.normal(m1v,s1v**2,(Dv,Lv))
        V = rd.normal(m2,s2**2,(D,C))
    
    eta = [Wv,Wu,V,mu,nu,sigma2,tau2]
    
    if disp:
        print("eta =",eta)
        print("thetas =",thetas)
        
    return eta, thetas

def noisy_data_spemix_1(eta,thetas,N,N_min=2,disp=False):
    '''Simule un jeu de données selon la mixture sur seulement certaines dimensions des espaces observé et latent de modèles linéaires Gaussiens simples.
    (Modèle (M.6.7)).
    
    Paramètres
    ----------
    eta : list,
        Liste de paramètres de la forme [Wv,Wu,mu,nu,sigma2,tau2], où :
            - Wv est une matrice de taille (Dv,Lv) dont les colonnes sont les axes principaux des dimensions "mixtes".
            - Wu est une matrice de taille (Du,Lu) dont les colonnes sont les axes principaux des dimensions "non-mixtes".
            - mu est un vecteur de taille Dv+Du, supposé être la moyenne des vecteurs observés.
            - nu est un vecteur de taille Lu, supposé être la moyenne des vecteurs latents dont la loi ne change pas selon le cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des vecteurs observés.
            - tau2 est un réel positif, supposé être la variance des vecteurs latents dont la loi ne change pas selon le cluster.
        On doit avoir Dv > Lv, et Du > Lu.
        
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu_k,sigma2_k], où :
            - mu_k est un vecteur de taille Lv, supposé être une moyenne locale de vecteurs latents dont la loi change selon le cluster.
            - sigma2_k est un réel positif, supposé être une variance locale de vecteurs latents dont la loi change selon le cluster.
           
    N : int,
        Nombre d'individus à simuler.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur 2 par défaut.
        N doit être strictement supérieur à K*N_min.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Zv : 2-D ndarray,
        Matrice de taille (N,Lv) dont les lignes sont les vecteurs latents dont la loi change selon les clusters.
    
    Zu : 2-D ndarray,
        Matrice de taille (N,Lu) dont les lignes sont les vecteurs latents dont la loi ne change pas selon les clusters.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,Dv+Du) dont les lignes sont les vecteurs observés obtenus.
    
    iota : 1-D ndarray,
        Vecteur de taille Dv+Du contenant Dv fois le nombre 1 et Du fois le nombre 0, où le nombre 1 signifie que la loi des observations sur l'axe correspondant diffère selon les clusters, et le nombre 0 signifie que la loi des observations sur l'axe correspondant ne diffère pas selon les clusters.
    
    Pour tout n entre 0 et N-1, si y, zv et zu sont les n-ièmes lignes respectives de Y, Zv et Zu,
    et R est la matrice ra_matrix(iota), alors y, zv et zu sont liés par la relation suivante :
    y = R.Concatenate(Wv.zv,Wu.zu) + mu + epsilon[n],
    et epsilon[n] est un vecteur gaussien centré, indépendant de zv et zu, de variance sigma2*Id.
    '''
    Wv, Wu, mu, nu, sigma2, tau2 = eta
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
    
    Zu = rd.normal(nu,tau2,(N,Lu))
    
    Y_prov = np.concatenate([Zv@np.transpose(Wv),Zu@np.transpose(Wu)],axis=1)
    iota_prov = np.concatenate([np.ones(Dv), np.zeros(Du)]).astype(int)
    
    noise = rd.normal(0,sigma2,(N,D))
    iota = rd.permutation(iota_prov)
    
    Y = drr.rearg(Y_prov,iota) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$\omega = $', omega)
        print('$\iota = $', iota)
    
    return Zv, Zu, omega, Y, iota

def noisy_data_spemix_2(eta,thetas,N,N_min=2,Sigma_X=None,disp=False):
    '''Simule un jeu de données selon la mixture sur seulement certaines dimensions des espaces observé et latent de modèles linéaires Gaussiens mixtes.
    (Modèle (M.6.8))
    
    Paramètres
    ----------
    eta : list,
        Liste de paramètres de la forme [Wv,Wu,V,mu,nu,sigma2,tau2], où :
            - Wv est une matrice de taille (Dv,Lv) dont les colonnes sont les axes principaux des dimensions "mixtes".
            - Wu est une matrice de taille (Du,Lu) dont les colonnes sont les axes principaux des dimensions "non-mixtes".
            - V est une matrice de taille (D,C) d'effets fixes.
            - mu est un vecteur de taille Dv+Du, supposé être la moyenne des vecteurs observés.
            - nu est un vecteur de taille Lu, supposé être la moyenne des vecteurs latents dont la loi ne change pas selon le cluster.
            - sigma2 est un réel positif, supposé être la variance du bruit des vecteurs observés.
            - tau2 est un réel positif, supposé être la variance des vecteurs latents dont la loi ne change pas selon le cluster.
        On doit avoir : Dv > Lv et Du > Lu, et C doit être inférieur ou égal à Dv+Du.
        
    thetas : list,
        Liste à K éléments, dont chacun est un ensemble de paramètres de la forme [mu_k,sigma2_k], où :
            - mu_k est un vecteur de taille Lv, supposé être une moyenne locale de vecteurs latents dont la loi change selon le cluster.
            - sigma2_k est un réel positif, supposé être une variance locale de vecteurs latents dont la loi change selon le cluster.
           
    N : int,
        Nombre d'individus à simuler.
    
    N_min : int, optional,
        Nombre minimal d'occurences de chaque entier entre 0 et K-1 dans le vecteur omega renvoyé.
        Prend comme valeur 2 par défaut.
        N doit être strictement supérieur à K*N_min.
    
    disp : float, optional,
        Si mis sur False, ne sert à rien.
        Si mis sur True, affiche les données simulées.
        Mis sur False par défaut. 
        
    Renvois
    -------
    Zv : 2-D ndarray,
        Matrice de taille (N,Lv) dont les lignes sont les vecteurs latents dont la loi change selon les clusters.
    
    Zu : 2-D ndarray,
        Matrice de taille (N,Lu) dont les lignes sont les vecteurs latents dont la loi ne change pas selon les clusters, indépendante de Zv.
    
    omega : 1-D ndarray,
        Vecteur de taille N dont, pour tout n entre 0 et N-1, la n-ième valeur est un entier entre 0 et K-1 indiquant le numéro du cluster auquel appartient le n-ième individu.
    
    Y : 2-D ndarray,
        Matrice de taille (N,Dv+Du) dont les lignes sont les vecteurs observés obtenus.
    
    X : 2-D ndarray,
        Matrice de taille (N,C) dont les lignes sont les vecteurs de covariables simulés, indépendante de Zv et Zu.
    
    iota : 1-D ndarray,
        Vecteur de taille Dv+Du contenant Dv fois le nombre 1 et Du fois le nombre 0, où le nombre 1 signifie que la loi des observations sur l'axe correspondant diffère selon les clusters, et le nombre 0 signifie que la loi des observations sur l'axe correspondant ne diffère pas selon les clusters.
    
    Pour tout n entre 0 et N-1, si y, zv et zu sont les n-ièmes lignes respectives de Y, Zv et Zu,
    et R est la matrice ra_matrix(iota), alors y, zv et zu sont liés par la relation suivante :
    y = R.Concatenate(Wv.zv,Wu.zu) + mu + epsilon[n],
    et epsilon[n] est un vecteur gaussien centré, indépendant de zv et zu, de variance sigma2*Id.
    '''
    Wv, Wu, V, mu, nu, sigma2, tau2 = eta
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
    
    Zu = rd.normal(nu,tau2,(N,Lu))
    
    Y_prov = np.concatenate([Zv@np.transpose(Wv),Zu@np.transpose(Wu)],axis=1)
    iota_prov = np.concatenate([np.ones(Dv), np.zeros(Du)]).astype(int)
    
    noise = rd.normal(0,sigma2,(N,D))
    iota = rd.permutation(iota_prov)
    
    if type(Sigma_X) == type(None) :
        X = rd.normal(0,1,(N,C))
    else :
        X = rd.multivariate_normal(np.zeros(C),Sigma_X,N)
    
    Y = drr.rearg(Y_prov,iota) + X@np.transpose(V) + noise + mu
    
    if disp:
        print('$Z = $', Z)
        print('$Y = $', Y)
        print('$Y = $', X)
        print('$\omega = $', omega)
        print('$\iota = $', iota)
    
    return Zv, Zu, omega, Y, X, iota