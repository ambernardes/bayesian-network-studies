import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
import networkx as nx
import pylab as plt


heartDisease = pd.read_csv('./data/dataset.csv')
heartDisease = heartDisease.replace('?',np.nan)

print('Sample instances from the dataset are given below')
print(heartDisease.head())

print('\n Attributes and datatypes')
print(heartDisease.dtypes)

model = BayesianModel([
    ('age','heartdisease'),
    ('gender','heartdisease'),
    ('exang','heartdisease'),
    ('cp','heartdisease'),
    ('heartdisease','restecg'),
    ('heartdisease','chol')
])

print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

print('\n Inferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)

print('\n 1. Probability of HeartDisease given evidence= restecg')
q1 = HeartDiseasetest_infer.query(variables=['heartdisease'], evidence={'restecg':1})
print(q1)

print('\n 2. Probability of HeartDisease given evidence= cp ')
q2 = HeartDiseasetest_infer.query(variables=['heartdisease'], evidence={'cp':2})
print(q2)

nx.draw(model, with_labels=True)
plt.savefig('model.png')
plt.close()