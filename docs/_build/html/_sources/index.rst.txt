.. LMLVQ Documentation documentation master file, created by
   sphinx-quickstart on Wed Sep 23 16:06:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LMLVQ documentation!
===============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Algorithm Description
=====================

Learning Vector Quantization(LVQ) is well-known for Supervised Vector Quantization. Large margin LVQ is to maximize the distance of sample margin or to maximize the distance between decision hyperplane and datapoints.

***********
Pseudo-code
***********

1. Get data with labels.
2. Initialize prototypes with labels.
3. Calculate Euclidean distance between datapoints and prototypes.
4. Extract minimum distance from datapoints to each prototypes with same label by using Euclidean distance.
5. Extract distances between datapoints and prototypes with different label by using Euclidean distance.
6. Compute the cost function:

.. math::
   E = \overbrace{\sum_{i=1}^{m} d_{i}^+}^{pull} + \frac{1}{2C} \cdot \overbrace{\sum_{i=1}^{m} \sum_{l \in I_i} ReLU(d_{i}^+ - d_{i,l} + \gamma)^2}^{push} 

7. Finally updates the prototypes:

.. math::
	w(t+1) = w(t) - \eta \frac{\partial E}{\partial w(t)}
	


Installation Requirements
=========================

Following are the basic requirements to run this program:

1. `python <www.python.org>`_ with minimum version 3.8.
2. `numpy <https://numpy.org/install/>`_ with minimum version 1.19.0.
3. `matplotlib <https://matplotlib.org/users/installing.html>`_.
4. `Scikit Learn <https://scikit-learn.org/stable/>`_ .
5. Ubuntu required version 20.04. 

**********
Execuation
**********

The execuation of the program required following steps:

1. Copy both files in a folder.
2. Go to folder and run the **python3 lmlvq_call.py** command.


Examples
========

These graph represents the training of model. On the left side it shows the datapoints with different classes and these are represented by set of prototypes with different classes. On the right side it shows the error which is minimizing on every iteration.

************
Iris Dataset
************

.. image:: images/graph.png
	:width: 400

************************
Wine Recognition Dataset
************************

.. image:: images/graph1.png
	:width: 400



Classes and Functions
=====================

*class*\ ``lmlvq_numpy.``\ **LMLVQ**\ (*prototype\_per\_class*)
	
	Large margin LVQ is to maximize the distance of sample margin or to maximize the distance between decision hyperplane and data point.
	
	**Attributes:**
	
		prototype_per_class:
    		
    			The number of prototypes per class to be learned.
    	

    	**normalization**\ (*input\_value*)
    		
    		Normalize the data between range 0 and 1.
    		
    		**Parameters:**
    		
    			input_value:
                		
                		A n x m matrix of input data.
                
                **Return:**
    		
    			normalized_data:
                		
                		A n x m matrix with values between 0 and 1.


	**prt**\ \ (*input\_data*, *data\_labels*, *prototype\_per\_class*)
    		
    		Calculate prototypes with labels either at mean or randomly depends on prototypes per class.
    		
    		**Parameters:**
    		
    			input\_value:
                		
                		A n x m matrix of datapoints.
		    	
		    	data\_labels:
				
				A n-dimensional vector containing the labels for each datapoint.
		    	
		    	prototypes per class:
				
				The number of prototypes per class to be learned. If it is equal to 1 then prototypes assigned at mean position else it assigns randolmy.
                
                **Return:**
    		
    			prototype\_labels:
                		
                		A n-dimensional vector containing the labels for each prototype.
            		
            		prototypes:
                		
                		A n x m matrix of prototyes.for training.


    	**euclidean_dist**\ (*input\_data*, *prototypes*)\
    		
    		Calculate squared Euclidean distance between datapoints and prototypes.
    		
    		**Parameters:**
    		
    			input\_data:
                		
                		A n x m matrix of datapoints.
            		
            		prototpes:
                		
                		A n x m matrix of prototyes of each class.
                
                **Return:**
    		
    			A n x m matrix with Euclidean distance between datapoints and prototypes.
    			

    	**distance_plus**\ (*data\_labels*, *prototype\_labels*, *prototypes*, *eu\_dist*)\
    		
    		Calculate squared Euclidean distance between datapoints and prototypes with same labels.
    		
    		**Parameters:**
    		
    			data\_labels:
    			
                		A n-dimensional vector containing the labels for each datapoint.
                		
            		prototype\_labels:
            		
                		A n-dimensional vector containing the labels for each prototype.
                		
            		prototpes:
            		
                		A n x m matrix of prototyes of each class.
                		
            		eu\_dist:
            		
                		A n x m matrix with Euclidean distance between datapoints and prototypes.
                
                **Return:**
    		
    			d\_plus:
    			
                		A n-dimensional vector containing distance between datapoints and prototypes with same label.
            		
            		w\_plus:
            		
                		A m x n matrix of nearest correct matching prototypes.
            		
            		w\_plus\_index:
            		
                		A n-dimensional vector containing the indices for nearest prototypes to datapoints with same label.
    		

    	**d_il**\ \ (*data\_labels*, *prototype\_labels*, *prototypes*, *eu\_dist*)\
    		
    		Calculate squared Euclidean distance between datapoints and prototypes with different labels.
    		
    		**Parameters:**
    		
    			data\_labels:
    			
                		A n-dimensional vector containing the labels for each datapoint.
            		
            		prototype\_labels:
            		
                		A n-dimensional vector containing the labels for each prototype.
            		
            		prototpes:
                		
                		A n x m matrix of prototyes of each class.
            		
            		eu\_dist:
                		
                		A n x m matrix with Euclidean distance between datapoints and prototypes.
                
                **Return:**
    		
    			A n x m matrix with Euclidean distance between datapoints and prototypes of different labels.		


    	**cost_function**\ (*dplus*, *in\_dist*, *constant*, *margin*)\
    		
    		Calculate cost function of LMLVQ.
    		
    		**Parameters:**
    		
    			d\_plus:
                    		A n-dimensional vector containing distance between datapoints and prototypes with same label.
                	in\_dist:
                    		A n x m matrix with Euclidean distance between datapoints and prototypes of different labels.
                	Constant:
                    		The regularization constant.
                	margin:
                    		The margin parameter.
                
                **Return:**
    		
    			normalized_data:
                		
                		The value which is the sum of all local errors.


    	**w_update**\ (*input\_data*, *data\_labels*, *prototypes*, *w\_plus\_index*, *dplus*, *in\_dist*, *constant*, *margin*, *\_lr*)\
    		
    		Calculate the update of prototypes.
    		
    		**Parameters:**
    		
    			input\_data:
                    		
                    		A n x m matrix of datapoints.
                	
                	data\_labels:
                    		
                    		A n-dimensional vector containing the labels for each datapoint.
                	
                	prototpes:
                    		
                    		A n x m matrix of prototyes of each class.
                	
                	w\_plus\_index:
                    		
                    		A n-dimensional vector containing the indices for nearest prototypes to datapoints with same label.
                	
                	d\_plus:
                    		
                    		A n-dimensional vector containing distance between datapoints and prototypes with same label.
                	
                	in\_dist:
                    		
                    		A n x m matrix with Euclidean distance between datapoints and prototypes of different labels.
                	
                	Constant:
                    		
                    		The regularization constant.
                	
                	margin:
                    		
                    		The margin parameter.
                	
                	\_lr:
                    		
                    		Learning rate also called step size.
                
                **Return:**
    		
    			The result of updated prototypes after calculated the update of prototypes with same and different labels. 		
    		
    		
    	**relu**\ (*x\_value*)
    		
    		An activation function.
    		
    		**Parameters:**
    		
    			x\_value: 
    				
    				A n x m matrix of datapoints or any value.
                
                **Return:**
    		
    			It return the value if it is greater than 0 otherwise returns 0.
                		

    	**plot**\ (*input\_data*, *data\_labels*, *prototypes*, *prototype\_labels*)\
    		
    		Scatter plot of data and prototypes.
    		
    		**Parameters:**
    		
    			input\_data:
                		A n x m matrix of datapoints.
            		data\_labels:
                		A n-dimensional vector containing the labels for each datapoint.
            		prototpes:
                		A n x m matrix of prototyes of each class.
            		prototype\_labels: 
            			A n-dimensional vector containing the labels for each prototype.
                		    		

    	**fit**\ (*input\_data*, *data\_labels*, *learning\_rate*, *epochs*, *margin*, *constant*)
    		
    		TODO.
    		
    		**Parameters:**
    		
    			input\_data:
                		A m x n matrix of distances. Note that we have no preconditions for this matrix.
            		data\_labels:
                		A m dimensional label vector for the data points.
            		learning\_rate:
                		The step size.
            		epochs:
                		The maximum number of optimization iterations.
            		margin:
                		The margin parameter.
            		Constant:
                		The regularization constant.


    	**predict**\ (*input\_value*)
    		
    		redicts the labels for the data represented by the given test-to-training distance matrix. Each datapoint will be assigned to the closest prototype.
    		
    		**Parameters:**
    		
    			input\_value:
                		A n x m matrix of distances from the test to the training datapoints.
                
                **Return:**
    		
    			ylabel:
               	 	A n-dimensional vector containing the predicted labels for each datapoint.

