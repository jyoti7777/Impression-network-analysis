# Impression Network Analysis

## Description

This project develops a methodology to identify communities within a graph and locate bridge nodes that connect these communities. This is particularly significant in social networks where individuals often belong to multiple groups, serving as conduits for information dissemination. The approach leverages Python for data manipulation, graph creation, and visualization, focusing on enhancing our understanding of network dynamics and facilitating targeted interventions for information dissemination and network optimization.

## Key Responsibilities

- Utilized Python libraries such as pandas, networkx, numpy, and matplotlib for data manipulation, graph creation, and visualization.
- Cleaned and prepared data by reading from an Excel sheet and removing duplicates.
- Created a directed graph from the data, adding nodes and edges based on relationships described in the dataset.

## Methodology

### 1. Community Identification
- Employed the Label Propagation method to assign unique labels to each node.
- Propagated labels through the graph, allowing nodes to adopt the most common label among their neighbors until community saturation was achieved.

### 2. Boundary Node Detection
- Identified boundary nodes that connected to other communities, serving as transition points between different community clusters.

### 3. Bridge Node Localization
- Determined bridge nodes as those connected to the maximum number of communities outside their own.
- These nodes were pivotal in bridging disparate communities and facilitating communication and information flow across the network.

## Technical Implementation

- Implemented a random walk algorithm with teleportation to assign ranks to nodes and identify the top leader in the impression network.
- Created and analyzed adjacency matrices to identify potential missing links using linear algebra concepts.
- Applied matrix operations and thresholding to infer missing links and uncover hidden relationships within the network.

## Tools and Libraries

- **pandas:** For reading data from Excel sheets and performing data cleaning.
- **networkx:** For creating and analyzing graph structures.
- **numpy:** For performing matrix operations and linear algebra calculations.
- **matplotlib.pyplot:** For visualizing the graph.

## Results

- Successfully identified key communities and bridge nodes, enhancing the understanding of network dynamics.
- Enabled targeted interventions for information dissemination and network optimization by identifying pivotal nodes within the network.

## Conclusion

The fusion of linear algebra and graph theory in this project presented a potent method for inferring missing links in networks. By adapting techniques akin to the PageRank algorithm and analyzing adjacency matrices, we systematically identified potential missing connections. This approach provided a mathematically grounded framework for uncovering hidden relationships, deepening our understanding of network structures, and advancing the field of network analysis.
