# Large-Margin-LVQ
Large Margin Learning Vector Quantization (LMLVQ)

----------------------------------------------------------------------------------------------------------------------------------
Learning Vector Quantization(LVQ) is well-known for Supervised Vector Quantization. Large margin LVQ is to maximize the distance of sample margin or to maximize the distance between decision hyperplane and datapoints.

## Pseudo-code

1) Get data with labels.
2) Initialize prototypes with labels.
3) Calculate Euclidean distance between datapoints and prototypes.
4) Extract minimum distance from datapoints to each prototypes with same label by using Euclidean distance.
5) Extract distances between datapoints and prototypes with different label by using Euclidean distance.
6) Compute the cost function:

![equation](https://latex.codecogs.com/svg.latex?E&space;=&space;\overbrace{\sum_{i=1}^{m}d_{i}^+}^{pull}&space;+&space;\frac{1{2C}\cdot&space;\overbrace{\sum_{i=1}^{m} \sum_{l \in I_i} ReLU(d_{i}^+&space;-&space;d_{i,l}&space;+&space;\gamma)^2}^{push})
