import bnlearn
import pandas as pd
from scipy.stats import hypergeom

# Load titanic dataset
df = bnlearn.import_example(data='titanic')

print(df[['Survived','Sex']])

# Total number of samples
N=df.shape[0]
# Number of success in the population
K=sum(df['Survived']==1)
# Sample size/number of draws
n=sum(df['Sex']=='female')
# Overlap between female and survived
x=sum((df['Sex']=='female') & (df['Survived']==1))

print(x-1, N, n, K)

# Compute
P = hypergeom.sf(x, N, n, K)
P = hypergeom.sf(232, 891, 314, 342)

print(P)
# 3.5925132664684234e-60