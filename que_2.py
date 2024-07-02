import pandas as pd
import networkx as nx
import numpy as np
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

   
# Generate adjacency matrix
adj_matrix = nx.adjacency_matrix(G)

# Convert to NumPy array 
adj_matrix = adj_matrix.toarray()

# Write the adjacency matrix to a file 
np.savetxt('adj_matrix.txt', adj_matrix, fmt='%d')

#Create the transposed matrix too to be used later
transposed_adj_matrix = np.transpose(adj_matrix) 


n, m = transposed_adj_matrix.shape


def find_link(A,i,j,X,M): 
    X=np.transpose(X)
    link = np.dot(X,M)
    if(link.item()<=0):
        x=0
    else:
        x=1
    return x


def find_unknowns_1(A, i, j):
    try:
        if ((i-1)<=0) or ((j-1)<=0):
            raise CustomError("Custom error occurred.")
        # Select submatrix of A and column matrix X
        A_sub = A[:i-1, :j-1]
        B = np.array([A[i, 0:j-1]])  # B is the transpose of the ith row of A from indices 1 to j-1
        A_sub =np.transpose(A_sub)
        B=np.transpose(B)
        X, residuals, rank, s = np.linalg.lstsq(A_sub, B, rcond=None)
        M = np.array([A[0:i-1, j]])
        M =M.reshape(-1, 1)
        return find_link(A,i,j,X,M)
        
    except (np.linalg.LinAlgError, IndexError, CustomError) as e:
        #Exception occurred in find_unknowns_1
        return 0


for i in range(n):
    for j in range(m):
        if transposed_adj_matrix[i, j] == 0 and adj_matrix[i, j] == 0 and i!=j:
            link = find_unknowns_1(transposed_adj_matrix, i, j)
            transposed_adj_matrix[i, j] = link
            adj_matrix[i, j] = link

np.savetxt('new_adj_matrix.txt',adj_matrix, fmt='%d')

name_to_index = {node: i for i, node in enumerate(G.nodes)}

# Add edges to the predefined graph based on the adjacency matrix
num_nodes = len(adj_matrix)
for i in range(num_nodes):
    for j in range(num_nodes):
        if adj_matrix[i][j] != 0:  # If non-zero entry in the adjacency matrix
            node_i = list(G.nodes())[i]  # Get node name corresponding to index i
            node_j = list(G.nodes())[j]  # Get node name corresponding to index j
            G.add_edge(node_i, node_j)

nx.draw(G, with_labels = True)
plt.show()
