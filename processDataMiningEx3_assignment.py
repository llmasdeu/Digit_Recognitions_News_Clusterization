#!/opt/local/bin/python2.7
# -*- coding: utf-8 -*-

#
# Mineria de Dades
# Pràctica 3: Aprendre a categoritzar imatges de dígits i classificació de grups de notícies
# Curs 2019-2020
#
# Autor: Lluís Masdeu
# Login: lluis.masdeu
#

import numpy as np
import scipy
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.neighbors
import sklearn.neural_network as neural_network
import sklearn.metrics as metrics
import sklearn.ensemble as ensemble
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.image as mpimg
import pandas as pd
import time

#
# Funció encarregada de dur a terme l'anàlisi del dataset dels dígits.
#
def digits_dataset_analysis():
    print '\t***** Anàlisi del dataset dels dígits *****\n'

    # Obtenim i analizem les dades del dataset
    X, Y = get_digits_dataset()

    # Obtenim els conjunts d'entrenament i de test sense normalitzar, normalitzarem i el percentatge de divisió utilitzats
    X_std, X_train, Y_train, X_test, Y_test, X_train_std, X_test_std, train_size, test_size = \
        separate_and_standardize_digits_data(X, Y)

    # Duem a terme la classificació amb l'algorisme de K veïns més propers
    # digits_dataset_k_neighbors(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 5)
    # digits_dataset_k_neighbors(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 10)
    # digits_dataset_k_neighbors(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 15)
    digits_dataset_k_neighbors(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 20)
    # digits_dataset_k_neighbors(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 25)
    # digits_dataset_k_neighbors(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 30)

    # Duem a terme la classficiació amb l'algorisme de xarxes neuronals
    # digits_dataset_neural_networks(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 10)
    # digits_dataset_neural_networks(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 20)
    # digits_dataset_neural_networks(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 30)
    # digits_dataset_neural_networks(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 40)
    # digits_dataset_neural_networks(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, 50)

    # Duem a terme la classificació amb l'algorisme AdaBoost
    digits_dataset_adaboost(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test)

#
# Funció encarregada de carregar i analitzar el dataset de les imatges amb els números.
# Retorna les dades i la seva descripció.
#
def get_digits_dataset():
    # Obtenim el dataset de les imatges amb els números
    # URL del dataset: http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    digits = sklearn.datasets.load_digits()

    # Mostrem els diferents camps de dades que té el dataset carregat
    # print digits.keys()

    # Separem les dades obtingudes en dades i la seva descripció
    X = digits.data
    Y = digits.target

    # Mostrem les dimensions de les dades obtingudes
    # print X.shape, Y.shape

    # Mostrem les dades obtingudes
    # print X, Y

    # Mostrem la descripció de les dades obtingudes
    # print digits.DESCR

    # Obtenim i mostrem les estadístiques bàsiques de les dades carregades
    print 'Classe X: La mitjana aritmètica de les dades és ', np.mean(X)
    print 'Classe Y: La mitjana aritmètica de les dades és ', np.mean(Y)
    print 'Classe X: La desviació típica de les dades és ', np.std(X)
    print 'Classe Y: La desviació típica de les dades és ', np.std(Y)
    print 'Classe X: Tenim ', X.shape[0], ' imatges de 8x8 (representats en tuples de 64)'
    print 'Classe Y: Tenim ', Y.shape[0], ' dades'

    # Reconstruïm la primera imatge del dataset, i la mostrem
    # image = np.reshape(X[0], (8, 8))
    # plt.imshow(image, interpolation='nearest')
    # plt.show()

    # Reconstruïm les primeres 9 imatges del dataset, i les mostrem en una graella
    # f = plt.figure()
    #
    # for x in range (9):
    #     f.add_subplot(3, 3, x + 1)
    #     image = np.reshape(X[x], (8, 8))
    #     plt.imshow(image, interpolation='nearest')
    #
    # plt.show()

    # Reconstruïm les primeres 10 imatges del dataset, i les mostrem en una graella
    # f = plt.figure()
    #
    # for x in range(10):
    #     f.add_subplot(2, 5, x + 1)
    #     image = np.reshape(X[x], (8, 8))
    #     plt.imshow(image, interpolation='nearest')
    #
    # plt.show()

    return X, Y

#
# Funció encarregada de dividir les dades en els conjunts d'entrenament i de test, i de normalitzar-los.
# Retorna els conjunts totals, d'entrenament i de test sense normalitzar, normalitzats, i el percentatge de divisió utilitzat.
#
def separate_and_standardize_digits_data(X, Y):
    # Definim la mida dels conjunts d'entrenament i de test de les dades
    train_size = 0.7
    test_size = 0.3

    # Dividim les dades en els conjunt d'entrenament i de test
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_size, train_size=train_size)

    # Mostrem les dimensions dels conjunts d'entrenament i de test de les dades
    # print X_train.shape, Y_train.shape
    # print X_test.shape, Y_test.shape

    # Mostrem les dades dels conjunts d'entrenament i de test de les dades
    # print X_train, Y_train
    # print X_test, Y_test

    # Normalitzem les dades per tal que estiguin centrades a 0 amb desviació típica 1
    # X_std = sklearn.preprocessing.scale(X)
    # X_train_std = sklearn.preprocessing.scale(X_train)
    # X_test_std = sklearn.preprocessing.scale(X_test)
    X_std = sklearn.preprocessing.StandardScaler().fit_transform(X)
    X_train_std = sklearn.preprocessing.StandardScaler().fit_transform(X_train)
    X_test_std = sklearn.preprocessing.StandardScaler().fit_transform(X_test)

    # Mostrem les dimensions de les dades d'entrenament i de test normalitzades
    # print X_train_std.shape, X_test_std.shape

    # Mostrem les dades d'entrenament i de test normalitzades
    # print X_train_std, X_test_std

    return X_std, X_train, Y_train, X_test, Y_test, X_train_std, X_test_std, train_size, test_size

#
# Funció encarregada de dur a terme els anàlisis a les dades de test amb l'algorisme de K Neighbors.
#
def digits_dataset_k_neighbors(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, n_neighbors):
    # Duem a terme la validació creuada amb les dades sense normalitzar mitjançant la tècnica PCA
    pca_cross_validation(X_train, Y_train, X_test, Y_test, n_neighbors, 'PCA - Cross validation')

    # Duem a terme la validació creuada amb les dades normalitzades mitjançant la tècnica PCA
    # pca_cross_validation(X_train_std, Y_train, X_test_std, Y_test, n_neighbors, 'PCA (Std) - Cross validation')

    # Duem a terme la validació creuada amb les dades sense normalitzar mitjançant la tècnica SVD-Truncated
    svd_truncated_cross_validation(X_train, Y_train, X_test, Y_test, n_neighbors, 'SVD-T - Cross validation')

    # Duem a terme la validació creuada amb les dades normalitzades mitjançant la tècnica SVD-Truncated
    # svd_truncated_cross_validation(X_train_std, Y_train, X_test_std, Y_test, n_neighbors, 'SVD-T (Std) - Cross validation')

#
# Funció encarregada de dur a terme la validació creuada mitjançant la tècnica PCA.
#
def pca_cross_validation(X_train, Y_train, X_test, Y_test, n_neighbors, title):
    # Definim el paràmetre de n_neighbors
    parameters = get_n_neighbors_parameter(n_neighbors)

    # Definim el classificador K-Nearest Neighbors
    knearest = sklearn.neighbors.KNeighborsClassifier()

    # Definim la estructura que ens servirà per desar els resultats
    gridsearch = sklearn.model_selection.GridSearchCV(knearest, parameters, cv=10, iid=True)

    # Definim els arrays buits que ens ajudaran a desar els resultats
    accuracy = []
    params = []
    means = []

    # Obtenim l'array amb les dimensions
    dimensions = get_dimensions_array()

    # Per cada n dimensions...
    for d in dimensions:
        # Fem la descomposició en d components
        pca = sklearn.decomposition.PCA(n_components=d)

        if d < 64:
            X_fit = pca.fit_transform(X_train)
            X_fit_atest = pca.transform(X_test)
        else:
            X_nl = X_train
            X_nl1 = X_test

        # Calculem i desem els resultats
        gridsearch.fit(X_fit, Y_train)
        result = compute_test(X_fit_atest, Y_test, gridsearch, 10)
        accuracy.append(result)
        means.append(np.mean(result))
        params.append(gridsearch.best_params_['n_neighbors'])

    # Mostrem els valors obtinguts
    # print accuracy, means, params
    print '[Validació creuada PCA] Precisió obtinguda (', n_neighbors, '): ', accuracy
    print '[Validació creuada PCA] Mitjanes obtingudes (', n_neighbors, '): ', means
    print '[Validació creuada PCA] Veïns més propers (', n_neighbors, '): ', params

    # Generem el gràfic amb els resultats
    draw_accuracy(means, dimensions, title)

#
# Funció encarregada de dur a terme la validació creuada mitjançant la tècnica SVD-Truncated.
#
def svd_truncated_cross_validation(X_train, Y_train, X_test, Y_test, n_neighbors, title):
    # Definim el paràmetre de n_neighbors
    parameters = get_n_neighbors_parameter(n_neighbors)

    # Definim el classificador K-Nearest Neighbors
    knearest = sklearn.neighbors.KNeighborsClassifier()

    # Definim la estructura que ens servirà per desar els resultats
    gridsearch = sklearn.model_selection.GridSearchCV(knearest, parameters, cv=10, iid=True)

    # Definim els arrays buits que ens ajudaran a desar els resultats
    accuracy = []
    params = []
    means = []

    # Obtenim l'array amb les dimensions
    dimensions = get_dimensions_array()

    # Per cada d dimensions...
    for d in dimensions:
        # Fem la descomposició en d components
        svd = sklearn.decomposition.TruncatedSVD(n_components=d)

        if d < 64:
            X_fit = svd.fit_transform(X_train)
            X_fit_atest = svd.transform(X_test)
        else:
            X_nl = X_train
            X_nl1 = X_test

        # Calculem i desem els resultats
        gridsearch.fit(X_fit, Y_train)
        result = compute_test(X_fit_atest, Y_test, gridsearch, 10)
        accuracy.append(result)
        means.append(np.mean(result))
        params.append(gridsearch.best_params_['n_neighbors'])

    # Mostrem els valors obtinguts
    # print accuracy, means, params
    print '[Validació creuada SVD-T] Precisió obtinguda (', n_neighbors, '): ', accuracy
    print '[Validació creuada SVD-T] Mitjanes obtingudes (', n_neighbors, '): ', means
    print '[Validació creuada SVD-T] Veïns més propers (', n_neighbors, '): ', params

    # Generem el gràfic amb els resultats
    draw_accuracy(means, dimensions, title)

#
# Funció encarregada de generar els paràmetres que emprarem per a calcular la validació creuada.
#
def get_n_neighbors_parameter(n_neighbors):
    k = np.arange(n_neighbors) + 1
    parameters = {'n_neighbors': k}

    return parameters

#
# Funció encarregada de generar l'array amb les dimensions que emprarem per dur a terme la validació creuada.
#
def get_dimensions_array():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#
# Funció encarregada de calcular les puntuacions d'encert.
#
def compute_test(X_test, Y_test, clf, cv):
    KFolds = sklearn.model_selection.KFold(n_splits=cv)
    scores = []

    for i, j in KFolds.split(X_test, Y_test):
        test_set = X_test[j]
        test_labels = Y_test[j]
        scores.append(metrics.accuracy_score(test_labels, clf.predict(test_set)))

    return scores

#
# Funció encarregada de mostrar en un diagrama de barres els resultats obtinguts amb la validació creuada.
#
def draw_accuracy(means, dimensions, title):
    # Preparem les dades
    means_def = np.asarray(means)
    pos = np.arange(len(dimensions))

    # Generem el gràfic amb les dades
    plt.bar(pos, np.array(means_def))
    plt.xticks(pos, dimensions)
    plt.title(title)
    plt.xlabel('Dimensions')
    plt.ylabel('Accuracy')
    plt.show()

#
# Funció encarregada de dur a terme l'anàlisi amb l'algorisme de xarxes neuronals.
#
def digits_dataset_neural_networks(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test, iterations):
    # Prediem les etiquetes de les dades de test a partir de les dades sense normalitzar
    score_train, score_test = neural_networks_predict_and_compute_accuracy(X_train, Y_train, X_test, Y_test, iterations)

    # Mostrem el percentatge de certesa de l'algorisme amb les dades sense normalitzar
    print 'La predicció amb xarxes neuronals (', iterations, ' iteracions), a partir de les dades sense normalitzar, de les dades d\'entrenament és del ', score_train, ' %'
    print 'La predicció amb xarxes neuronals (', iterations, ' iteracions), a partir de les dades sense normalitzar, de les dades de test és del ', score_test, ' %'

    # Prediem les etiquetes de les dades de test a partir de les dades normalitzades
    score_train, score_test = neural_networks_predict_and_compute_accuracy(X_train_std, Y_train, X_test_std, Y_test, iterations)

    # Mostrem el percentatge de certesa de l'algorisme amb les dades normalitzades
    print 'La predicció amb xarxes neuronals (', iterations, ' iteracions), a partir de les dades normalitzades, de les dades d\'entrenament és del ', score_train, ' %'
    print 'La predicció amb xarxes neuronals (', iterations, ' iteracions), a partir de les dades normalitzades, de les dades de test és del ', score_test, ' %'

#
# Funció encarregada de predir i calcular la certesa de l'algorisme amb xarxes neuronals.
# Retorna la puntuació de certesa de l'algorisme amb les dades d'entrenament i amb les dades de test.
#
def neural_networks_predict_and_compute_accuracy(X_train, Y_train, X_test, Y_test, iterations):
    # Creem la xarxa neuronal
    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(100, 100,), activation='relu', solver='sgd',
                                      learning_rate='constant', learning_rate_init=0.02, n_iter_no_change=iterations)

    # Entrenem la xarxa neuronal amb les dades d'entrenament
    mlp.fit(X_train, Y_train)

    # Prediem els resultats d'entrenament
    Y_pred = mlp.predict(X_train)

    # Mostrem les etiquetes d'entrenament, les etiquetes predites, i la diferència entre les dues
    # print Y_train, Y_pred, Y_train - Y_pred

    # Calculem la puntuació de certesa de l'algorisme
    score_train = metrics.accuracy_score(Y_train, Y_pred)

    # Mostrem la puntuació de certesa de l'algorisme
    # print score_train

    # Prediem els resultats de test
    Y_pred = mlp.predict(X_test)

    # Mostrem les etiquetes de test, les etiquetes predites, i la diferència entre les dues
    # print Y_test, Y_pred, Y_test - Y_pred

    # Calculem la puntuació de certesa de l'algorisme
    score_test = metrics.accuracy_score(Y_test, Y_pred)

    # Mostrem la puntuació de certesa de l'algorisme
    # print score_test

    return score_train * 100, score_test * 100

#
# Funció encarregada de dur a terme l'anàlisi amb l'algorisme AdaBoost.
#
def digits_dataset_adaboost(X_train, X_train_std, Y_train, X_test, X_test_std, Y_test):
    # Prediem les etiquetes de les dades de test a partir de les dades sense normalitzar
    score_train, score_test = adaboost_classify_and_compute_accuracy(X_train, Y_train, X_test, Y_test)

    # Mostrem el percentatge de certesa de l'algorisme amb les dades sense normalitzar
    print 'La predicció amb AdaBoost, a partir de les dades sense normalitzar, de les dades d\'entrenament és del ', score_train, ' %'
    print 'La predicció amb AdaBoost, a partir de les dades sense normalitzar, de les dades de test és del ', score_test, ' %'

    # Prediem les etiquetes de les dades de test a partir de les dades normalitzades
    score_train, score_test = adaboost_classify_and_compute_accuracy(X_train_std, Y_train, X_test_std, Y_test)

    # Mostrem el percentatge de certesa de l'algorisme amb les dades normalitzades
    print 'La predicció amb AdaBoost, a partir de les dades normalitzades, de les dades d\'entrenament és del ', score_train, ' %'
    print 'La predicció amb AdaBoost, a partir de les dades normalitzades, de les dades de test és del ', score_test, ' %'

#
# Funció encarregada de predir i calcular la certesa de l'algorisme amb AdaBoost.
# Retorna la puntuació de certesa de l'algorisme amb les dades d'entrenament i les dades de test.
#
def adaboost_classify_and_compute_accuracy(X_train, Y_train, X_test, Y_test):
    # Creem el classificador AdaBoost
    ab = ensemble.AdaBoostClassifier(sklearn.tree.DecisionTreeClassifier(max_depth=10), n_estimators=200)

    # Entrenem el classificador amb les dades d'entrenament
    ab.fit(X_train, Y_train)

    # Prediem les etiquetes de les dades d'entenament
    Y_pred = ab.predict(X_train)

    # Mostrem les etiquetes d'entrenament, les etiquetes predites, i la diferència entre les dues
    # print Y_train, Y_pred, Y_train - Y_pred

    # Calculem la puntuació de certesa de l'algorisme
    score_train = metrics.accuracy_score(Y_train, Y_pred)

    # Mostrem la puntuació de certesa de l'algorisme
    # print score_train

    # Prediem les etiquetes de les dades de test
    Y_pred = ab.predict(X_test)

    # Mostrem les etiquetes de test, les etiquetes predites, i la diferència entre les dues
    # print Y_test, Y_pred, Y_test - Y_pred

    # Calculem la puntuació de certesa de l'algorisme
    score_test = metrics.accuracy_score(Y_test, Y_pred)

    # Mostrem la puntuació de certesa de l'algorisme
    # print score_test

    return score_train * 100, score_test * 100

#
# Funció principal del programa.
#
if __name__ == '__main__':
    # Duem a terme l'anàlisi del dataset dels dígits
    digits_dataset_analysis()

    exit(0)
