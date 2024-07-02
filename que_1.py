import pandas as pd
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

class CustomError(Exception):
    pass
# Read data from Excel sheet
df = pd.read_excel('impressions.xlsx')

# Drop duplicate rows from the DataFrame
df.drop_duplicates(inplace=True)

# Create an empty directed graph
G = nx.DiGraph()

# Add nodes from the first column
nodes = df.iloc[:, 0]  # Selecting the first column
unique_nodes = nodes.unique()
G.add_nodes_from(unique_nodes)  # Add unique nodes


# Add edges from each row
for _, row in df.iterrows():
    source_node = row[0]  # Source node is in the first column
    for target_node in row[1:]:
        if pd.notnull(target_node):  # Check if the cell is not empty
            G.add_edge(source_node, target_node)

nx.draw(G, with_labels = True)
plt.show()                          # Display the final graph G using matplotlib


def rank_assign(G):
 points = 10000000   # Total initial points/coins to be distributed among the nodes 
 points_box = {}  
 for node in G.nodes():
        points_box[node] = 0   # Initializing the points of all the nodes in G to 0

 random_node = random.choice(list(G.nodes()))  
 while(points >= 0 ):
        adjacency_list = list(G.neighbors(random_node))
        if adjacency_list:
            neighbour = random.choice(adjacency_list)
        else:
            neighbour = random.choice(list(set(G.nodes()) - {random_node}))   # If the current node does not have a neighbour then choose any other node at random, the current node is removed from the graph to avoid choosing the same node again
        points_box[neighbour]=points_box[neighbour]+1       # increment the points of the randomly chosen neighbour
        points=points-1                                    # decrement the Total points by 1
        random_node = neighbour                            # make the neighbour the new random_node
 rank_1 = max(points_box, key=points_box.get)              # The top leader is the one with the maximum points after uniformity has been achieved 
 
 print("Top  leader :",rank_1)   
 return points_box

rank_assign(G)

