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

![equation](http://www.sciweavers.org/tex2img.php?eq=1%2Bsin%28mc%5E2%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)
