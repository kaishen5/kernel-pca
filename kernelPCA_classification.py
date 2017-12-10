# -*- coding: utf-8 -*-
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.kernel_approximation import Nystroem
filepath = 'C:/Users/kaish/Downloads/kPCA_v3.1/kPCA_v3.1/demo2/YaleFaceData.mat'
arrays = {}
f = h5py.File(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)


test_t=arrays['test_t']
test_x=np.transpose(arrays['test_x'])
train_t=arrays['train_t']
y_t=np.squeeze(test_t, axis=1)
y_tr=np.squeeze(train_t, axis=1)
train_x=np.transpose(arrays['train_x'])


y_tr[y_tr==-1]=0
y_t[y_t==-1]=0

gamma=(2**exp for exp in range(-50, 10, 2))


arrays2={}

for ga in gamma:

        scikit_kpca = KernelPCA(n_components=9, kernel='rbf', gamma=ga)
        
        X_skernpca=scikit_kpca.fit_transform(train_x)
        
        X_skernpca_t =scikit_kpca.transform(test_x)
        
        '''
        
        plt.figure(figsize=(10,8))
        plt.scatter(X_skernpca[y_tr==0, 0], X_skernpca[y_tr==0, 1], color='red', alpha=0.5)
        plt.scatter(X_skernpca[y_tr==1, 0], X_skernpca[y_tr==1, 1], color='blue', alpha=0.5)
        
        
        
        ax = plt.axes(projection='3d')
        ax.scatter(X_skernpca[y_tr==0, 0], X_skernpca[y_tr==0, 1], X_skernpca[y_tr==0, 2], color='red', alpha=0.5)
        ax.scatter(X_skernpca[y_tr==1, 0], X_skernpca[y_tr==1, 1], X_skernpca[y_tr==1, 2], color='blue', alpha=0.5)
        
        '''
       
        
        clf= LinearDiscriminantAnalysis()
      
        scores = cross_val_score(clf, X_skernpca, y_tr,cv=5)
        
        arrays2[math.log(ga,2)]=np.mean(scores)
        
        
lists = sorted(arrays2.items()) 

x, y = zip(*lists) 

plt.plot(x, y)
plt.show()      


scikit_kpca = KernelPCA(n_components=9, kernel='rbf', gamma=2**max(arrays2, key=arrays2.get))
X_skernpca=scikit_kpca.fit_transform(train_x)
X_skernpca_t =scikit_kpca.transform(test_x)





plt.figure(figsize=(10,8))
plt.scatter(X_skernpca[y_tr==0, 0], X_skernpca[y_tr==0, 1], color='red', alpha=0.5)
plt.scatter(X_skernpca[y_tr==1, 0], X_skernpca[y_tr==1, 1], color='blue', alpha=0.5)
plt.show()


ax = plt.axes(projection='3d')
ax.scatter(X_skernpca[y_tr==0, 0], X_skernpca[y_tr==0, 1], X_skernpca[y_tr==0, 2], color='red', alpha=0.5)
ax.scatter(X_skernpca[y_tr==1, 0], X_skernpca[y_tr==1, 1], X_skernpca[y_tr==1, 2], color='blue', alpha=0.5)
plt.show()



scikit_pca = PCA(n_components=9)        
X_pca = scikit_pca.fit_transform(train_x)
X_pca_t =scikit_pca.transform(test_x)


X_effi=Nystroem(kernel="rbf", gamma=2**(-30), n_components=50)
X_pc=X_effi.fit_transform(train_x)
X_pc_1=X_pc[:,0:9]
X_pc_t =X_effi.transform(test_x)
        
        

clf= LinearDiscriminantAnalysis()
clf.fit(X_skernpca, y_tr)
error1=1-accuracy_score(y_t,clf.predict(X_skernpca_t))

print(error1)


clf2 = LinearDiscriminantAnalysis()
clf2.fit(X_pca, y_tr)
error2=1-accuracy_score(y_t,clf2.predict(X_pca_t))

print(error2)


clf3 = LinearDiscriminantAnalysis()
clf3.fit(X_pc_1, y_tr)
error3=1-accuracy_score(y_t,clf3.predict(X_pc_t[:,0:9]))

print(error3)







image1=train_x[0,].reshape(168,192)

image2=train_x[58,].reshape(168,192)

from PIL import Image



plt.imshow(np.matrix.transpose(image1), cmap='gray')
plt.show()


plt.imshow(np.matrix.transpose(image2), cmap='gray')
plt.show()     



