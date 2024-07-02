
import pandas as pd
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

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
 return points_box


# Generate adjacency matrix
adj_matrix = nx.adjacency_matrix(G)

# Convert to NumPy array 
adj_matrix = adj_matrix.toarray()

# Write the adjacency matrix to a file
np.savetxt('adj_matrix.txt', adj_matrix, fmt='%d')

points_box=rank_assign(G)

# Step 1: Identify the communities
def label_propagation(G):
    # Initialize each node with a unique label
    for node in G.nodes():
        G.nodes[node]['label'] = node
    while True:
        # Shuffle the nodes to avoid bias in label propagation order
        nodes = list(G.nodes())
        random.shuffle(nodes)
        
        # Flag to track label changes
        labels_changed = False
        
        # Update labels based on neighbor labels
        for node in nodes:
            neighbor_labels = [G.nodes[neighbor]['label'] for neighbor in G.neighbors(node)]
            if neighbor_labels:
                most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
                # If all labels occur once, choose the label of the neighbor with the maximum rank value using the points_box dictionary as returned by rank_assign(G) function
                if neighbor_labels.count(most_common_label) == 1:
                    max_rank_neighbor = max(G.neighbors(node), key=lambda x: points_box.get(x, 0))
                    most_common_label = G.nodes[max_rank_neighbor]['label']
                if G.nodes[node]['label'] != most_common_label:
                    G.nodes[node]['label'] = most_common_label
                    labels_changed = True
        
        # Check for convergence
        if not labels_changed:
            break
    # Create a dictionary to store community memberships
    communities = {}
    for node in G.nodes():
        label = G.nodes[node]['label']
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    
    return communities

# Apply Label Propagation Algorithm
communities = label_propagation(G)
print(communities)


# Step 2: Determine boundary nodes
boundary_nodes = defaultdict(set)  # Dictionary to store boundary nodes for each community
for community_label, community_nodes in communities.items():
    for node in community_nodes:
        # Check if the node has neighbors outside its community
        if any(neighbor not in community_nodes for neighbor in G.neighbors(node)):
            boundary_nodes[community_label].add(node)

# Step 3: Find bridge nodes (nodes that connect to the maximum number of communities)
bridge_nodes = set()
for community_label, nodes in boundary_nodes.items():
    if len(nodes) == 1:  # If there's only one boundary node in the community, add it as a bridge node
        bridge_nodes.update(nodes)
    else:
        max_connections = 0
        bridge_node = None
        for node in nodes:
            # Count the number of communities connected to the current node
            connected_communities = {label for neighbor in G.neighbors(node)
                                            for label, community_nodes in boundary_nodes.items()
                                            if neighbor in community_nodes}
            num_connections = len(connected_communities)
            if num_connections > max_connections:
                bridge_node = node
                max_connections = num_connections
        if bridge_node is not None:
            bridge_nodes.add(bridge_node)

print("Bridge nodes connecting to the most communities:", bridge_nodes)
for node in bridge_nodes:
    connected_communities = set()
    
    # Add the node's own community label
    for label, community_nodes in communities.items():
        if node in community_nodes:
            connected_communities.add(label)
    
    # Add the communities of the node's neighbors
    for neighbor in G.neighbors(node):
        for label, community_nodes in communities.items():
            if neighbor in community_nodes:
                connected_communities.add(label)
    
    print("Node", node, "is connected to the following communities:", connected_communities)

