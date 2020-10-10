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

<img src="http://latex.codecogs.com/svg.latex?E&space;=&space;\overbrace{\sum_{i=1}^{m}&space;d_{i}^&plus;}^{pull}&space;&plus;&space;\frac{1}{2C}&space;\cdot&space;\overbrace{\sum_{i=1}^{m}&space;\sum_{l&space;\in&space;I_i}&space;ReLU(d_{i}^&plus;&space;-&space;d_{i,l}&space;&plus;&space;\gamma)^2}^{push}&space;" title="http://latex.codecogs.com/svg.latex?E = \overbrace{\sum_{i=1}^{m} d_{i}^+}^{pull} + \frac{1}{2C} \cdot \overbrace{\sum_{i=1}^{m} \sum_{l \in I_i} ReLU(d_{i}^+ - d_{i,l} + \gamma)^2}^{push} " />
