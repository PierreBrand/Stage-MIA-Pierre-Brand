'''====================================
    Ensemble de modules créés par Pierre
    ====================================
    
    ======================================= ============================================================
    Nécessite les bibliothèques suivantes :
    --------------------------------------- ------------------------------------------------------------
    numpy
    matplotlib
    numpy.random
    numpy.linalg
    pandas
    sklearn.covariance
    sklearn.cluster
    sklearn.metrics.cluster
    itertools
    =============================== ====================================================================
        
    =============================== ====================================================================
    Contient les modules suivants :
    ------------------------------- --------------------------------------------------------------------
    maybe_useful                    Fonctions peut-être utiles.
    for_clus                        Fonctions utiles pour le clustering.
    single_fa                       Fonctions d'estimation pour modèles simples.
    clus_funs                       Fonctions de clustering.
    data_rnr                        Fonctions pour reconstruire et représenter les données.
    for_fs                          Fonctions utiles pour la sélection de variables.
    mixed_fa                        Fonctions d'estimation pour mixtures de modèles.
    simulate                        Fonctions pour simuler des paramètres et des données.
    fs_funs                         Fonctions de sélection de variables.
    =============================== ====================================================================
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

print("Voulez-vous utiliser les abréviations automatiques ? Si oui, répondez 'oui'.")
answer = input()

if answer == 'oui':
    
    import maybe_useful as mbu
    print("Fonctions peut-être utiles : mbu")
    import for_clus as ufc
    print("Fonctions utiles pour le clustering : ufc")
    import single_fa as sfa
    print("Fonctions d'estimation pour modèles simples : sfa")
    import clus_funs as clf
    print("Fonctions de clustering : clf")
    import data_rnr as drr
    print("Fonctions pour reconstruire ou représenter les données : drr")
    import for_fs as ufs
    print("Fonctions utiles pour la sélection de variables : ufs")
    import mixed_fa as mfa
    print("Fonctions d'estimation pour mixtures de modèles : mfa")
    import simulate as sim
    print("Fonctions de simulation de jeux de données : sim")
    import fs_funs as fsf
    print("Fonctions de sélection de variables : fsf")

else :
    
    import maybe_useful
    print("Fonctions peut-être utiles : maybe_useful")
    import for_clus
    print("Fonctions utiles pour le clustering : for_clus")
    import single_fa
    print("Fonctions d'estimation pour modèles simples : single_fa")
    import clus_funs
    print("Fonctions de clustering : clus_funs")
    import data_rnr
    print("Fonctions pour reconstruire ou représenter les données : data_rnr")
    import for_fs
    print("Fonctions utiles pour la sélection de variables : for_fs")
    import mixed_fa
    print("Fonctions d'estimation pour modèles mixtes : mixed_fa")
    import simulate
    print("Fonctions de simulation de jeux de données : simulate")
    import fs_funs
    print("Fonctions de sélection de variables : fs_funs")