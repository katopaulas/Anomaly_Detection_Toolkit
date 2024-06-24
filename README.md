# Anomaly detection toolkit

## Introduction
Basic functions for fast prototyping and exploration of imbalanced binary classification problems aka anomaly/novelty detection. The tools provided have the ability to transform to multi class problems.

The basic idea in any anomaly detection problem is to find the compressed space that expresses the distribution density function of the normal samples and use this information to find any outliers.

## Project Overview
This is a general purpose repository but for a simple run

1. Build a data curation object that handles the data in a manner of X,Y where X is the data to process and Y the anomaly labels. Should be in the format of (-1,1) or (0,1) where -1 or 0 means an anomaly.

2.  `main.py` represents a simple interface where after the ETL part will run the following algorithms to get a first generic sense:
	
	1. SGDOneClassSVM
	2. LocalOutlierFactor
	3. IsolationForest
	4. Simple Neural Net with class weights and Focal loss