import numpy as np
import pandas as pd
import scipy.linalg as la

data = pd.io.parsers.read_csv('./data/ratings.csv', delimiter=',')
movie_data = pd.io.parsers.read_csv('./data/movies.csv', delimiter=',')

#small test data (10users, 10 movies, no NaN entries)
dataTest = pd.io.parsers.read_csv('./data/ratingsTiny.csv', delimiter=',')
movie_dataTest = pd.io.parsers.read_csv('./data/MoviesTiny.csv', delimiter=',')

ratings_mat = np.ndarray(
    shape=(np.max(data.movieId.values), np.max(data.userId.values)),
    dtype=np.uint8)
ratings_mat[data.movieId.values-1, data.userId.values-1] = data.rating.values

#test data block
ratings_mat_test = np.ndarray(
    shape=(np.max(dataTest.movieId.values), np.max(dataTest.userId.values)),
    dtype=np.uint8)
ratings_mat_test[dataTest.movieId.values-1, dataTest.userId.values-1] = dataTest.rating.values

normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
                                                    
                                                    #test data block
normalised_mat_test = ratings_mat_test - np.asarray([(np.mean(ratings_mat_test, 1))]).T

A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)

#test data block
ATest = normalised_mat_test.T / np.sqrt(ratings_mat_test.shape[0] - 1)

print(np.shape(A))
print(np.shape(ATest)) #looks good

##############################################

#writing out svd by hand: Source MIT opencourseware https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm

#find V: eigenvectors of A transpose * A
AT_A = ATest.transpose()@ATest
w,v = np.linalg.eig(AT_A) #w eigenvalues, v eigenvectors
ncols = np.argsort(w)[::-1]

#now, apply Gram-Schmidt transformation to make orthonormal
def gram_schmidt(X):
    Q, R = np.linalg.qr(X) #can i take this shortcut?
    return Q

v = v[:,ncols] #sort descending
V = gram_schmidt(v).transpose() #now V is orthonormal eigenvectors of A transpose * A


#Sigma: diagonal matrix whose entries are the sqrt of eigenvalues of Atranspose*A / A*Atranspose

roots = np.sqrt(np.abs(w)) #what to do if eig vals are negative?
sorted = np.sort(roots)[::-1] #sort in descending order
Sigma = np.diag(sorted, k=0) #k = 0 specifices main diagonal, square roots of w

#U: uniquely determined by V. Ui = Vi@A/sigmai
i = 0
m = np.zeros(10)
for x in V:
  y = 1/(sorted[i])*(ATest@x)
  if(i == 0):
    m = y
  else:
    m = np.c_[m, y]
  i+=1

U = m

#now,theoretically, have all three components, let's try to reconstruct:

ATestrec = U@Sigma@V

print(ATest)
print(ATestrec)

#for idx, x in np.ndenumerate(ATest):
  #if(x != ATestrec[idx]):
     # print(idx, x, ATestrec[idx])
