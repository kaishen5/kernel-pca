# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:31:08 2017

@author: kashen
"""

import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import skimage
from KPCA_self import kPCA
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.decomposition import KernelPCA

results=[]
from sas7bdat import SAS7BDAT
with SAS7BDAT('pricedata_17.sas7bdat') as f:
    for row in f:
        pass
        

saledata=pd.read_sas('pricedata_17.sas7bdat')
saledata=saledata['sale']


#f = lambda x: x.sort_values(ascending=True).reset_index(drop=True)
#sale_t=saledata.groupby(['product']).sale.apply(f).unstack()
#sm=pd.DataFrame.as_matrix(sale_t)
#
#
#grouped=saledata.groupby('product').apply(np.matrix.transpose)




saledata_prod1=saledata.iloc[:60]
saledata_prod1.plot(x='index',y='sale')

sm=pd.DataFrame.as_matrix(saledata)
sm=sm[0:60]
sm=np.reshape(sm, (60,1))

o1=KernelPCA(n_components=10, kernel="rbf",fit_inverse_transform=True)
trans=o1.fit_transform(sm)
t1=o1.inverse_transform(trans)

plt.plot(t1)




