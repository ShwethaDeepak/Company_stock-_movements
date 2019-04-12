#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:04:42 2018

@author: swetu
"""
# training day(that is the dollar difference between opening and closing . )
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
df = pd.read_csv('company-stock-movements-2010-2015-incl.csv')


normalize = Normalizer()
KMeans = KMeans(n_clusters = 10)

from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(normalize,KMeans)
movements = pd.read_csv('company-stock-movements-2010-2015-incl.csv')
movements_features = movements.iloc[:,1:965].values
companies = movements.iloc[:,0]

labels = pipeline.fit_predict(movements_features)
df = pd.DataFrame({'labels':labels,'companies': companies})
print(df.sort_values('labels'))

