
import bnlearn as bn
# Load sprinkler dataset
df = bn.import_example('sprinkler')
# The edges can be created using the available variables.
print(df.columns)
# ['Cloudy', 'Sprinkler', 'Rain', 'Wet_Grass']

# Define the causal dependencies based on your expert/domain knowledge.
# Left is the source, and right is the target node.
edges = [('Cloudy', 'Sprinkler'),
         ('Cloudy', 'Rain'),
         ('Sprinkler', 'Wet_Grass'),
         ('Rain', 'Wet_Grass')]


# Create the DAG
DAG = bn.make_DAG(edges)

# Plot the DAG. This is identical as shown in Figure 3
bn.plot(DAG)

# Print the Conditional probability Tables
bn.print_CPD(DAG)
# [bnlearn] >No CPDs to print. Tip: use bnlearn.plot(DAG) to make a plot.
# This is correct, we did not learn any CPTs yet! We only defined the graph without defining any probabilities.

# Parameter learning on the user-defined DAG and input data using maximumlikelihood
model_mle = bn.parameter_learning.fit(DAG, df, methodtype='maximumlikelihood')

# Print the learned CPDs
bn.print_CPD(model_mle)


# Parameter learning on the user-defined DAG and input data using bayesianestimation 
model_bayes = bn.parameter_learning.fit(DAG, df, methodtype='bayes')

# Print the learned CPDs
bn.print_CPD(model_bayes)