import bnlearn as bn

# Load sprinkler dataset
df = bn.import_example('sprinkler')

# Define the causal dependencies based on your expert/domain knowledge.
# Left is the source, and right is the target node.
edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]

# Create the DAG
DAG = bn.make_DAG(edges)

# Parameter learning on the user-defined DAG and input data using Bayes to estimate the CPTs
model = bn.parameter_learning.fit(DAG, df, methodtype='bayes')
# bn.print_CPD(model)

# How probable is it to have wet grass given the sprinkler is off? 
q1 = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Sprinkler':0})
print(q1.df)

# How probable is it to have Rain given the sprinkler is off e its clouldy? 
q2 = bn.inference.fit(model, variables=['Rain'], evidence={'Sprinkler':0, 'Cloudy':1})
print(q2.df)

# Inferences with two or more variables can also be made such as:
# How probable is it to have Rain and wet grass given the sprinkler is off e its clouldy? 
q3 = bn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})
print(q3.df)