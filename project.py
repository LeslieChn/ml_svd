#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy.linalg as la


def main():
    data = pd.io.parsers.read_csv('./data/ratings.csv', delimiter=',')
    movie_data = pd.io.parsers.read_csv('./data/movies.csv', delimiter=',')

    # small test data (10users, 10 movies, no NaN entries)
    dataTest = pd.io.parsers.read_csv('./data/ratingsTiny.csv', delimiter=',')
    movie_dataTest = pd.io.parsers.read_csv(
        './data/MoviesTiny.csv', delimiter=',')

    ratings_mat = np.ndarray(
        shape=(np.max(data.movieId.values), np.max(data.userId.values)),
        dtype=np.uint8)
    ratings_mat[data.movieId.values-1,
                data.userId.values-1] = data.rating.values

    # test data block
    ratings_mat_test = np.ndarray(
        shape=(np.max(dataTest.movieId.values),
               np.max(dataTest.userId.values)),
        dtype=np.uint8)
    print("shape:", ratings_mat_test.shape)
    ratings_mat_test[dataTest.movieId.values-1,
                     dataTest.userId.values-1] = dataTest.rating.values

    normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

    # test data block
    normalised_mat_test = ratings_mat_test - \
        np.asarray([(np.mean(ratings_mat_test, 1))]).T

    A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)

    # test data block
    ATest = normalised_mat_test.T / np.sqrt(ratings_mat_test.shape[0] - 1)

    print(np.shape(A))
    print(ATest[0])
    # print(np.shape(ATest))  # looks good
    u1, s1, v1 = SVD(ATest)
##############################################

# writing out svd by hand: Source MIT opencourseware https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm

# Gram-Schmidt transformation to make orthonormal


def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q


def SVD(A):
    # find V: eigenvectors of A transpose * A
    AT_A = A.T@A
    w, v = np.linalg.eig(AT_A)  # w eigenvalues, v eigenvectors
    ncols = np.argsort(w)[::-1]

    v = v[:, ncols]  # sort descending
    # now V is orthonormal eigenvectors of A transpose * A
    V = gram_schmidt(v).T

    # Sigma: diagonal matrix whose entries are the sqrt of eigenvalues of Atranspose*A / A*Atranspose

    roots = np.sqrt(np.abs(w))  # what to do if eig vals are negative?
    sorted = np.sort(roots)[::-1]  # sort in descending order
    # k = 0 specifices main diagonal, square roots of w
    Sigma = np.diag(sorted, k=0)

    # U: uniquely determined by V. Ui = Vi@A/sigmai
    i = 0
    m = np.zeros(10)
    for x in V:
        y = 1/(sorted[i])*(x@A)
        if (i == 0):
            m = y
        else:
            m = np.c_[m, y]
        i += 1

    U = m
    return U, Sigma, V


# if Anxm : U nxn, Sigma nxm, V mxm
# can reduce Sigma to c greatest singular vals (Symeonidis P. Matrix and Tensor Factorization Tech for Recommender Systems 2017 p 38)
# so: Anxm : U nxc, Sigma cxc, V cxm
# how to choose c? book: maintain 83% of original vals


def truncate(U, S, V, c):  # truncates U,S,Vt to nxc, cxc, cxm
    UNew = U[:U.shape[0], :c]
    SNew = S[:c, :c]
    VNew = V[:c, :V.shape[1]]
    return UNew, SNew, VNew


# print(truncate(u1,s1,v1,30))
if __name__ == "__main__":
    main()
