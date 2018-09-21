# Automated Diagnosis of Lung Cancer with 3D Convolutional Neural Networks

Pytorch code for the [Automated Prediction of Lung Cancer with 3D Convolutional Neural Networks](https://biomedicalcomputervision.uniandes.edu.co/index.php/research?id=32). Our cancer predictor obtained a ROC AUC of 0.913 and was ranked 1st place at the [ISBI 2018 Lung Nodule Malignancy Prediction challenge](https://bit.ly/2JPNnGS).

### Prerequisites

This code was implemented in Python 2.7.*, using PyTorch, Numpy, pandas, sklearn, scipy, skimage and dicom.

### Getting started

To run the code save the folder of each patient with the dicom files in the folder ./data/ISBI-deid-TRAIN/.
If the dataset from the [ISBI 2018 Lung Nodule Malignancy Prediction challenge](https://bit.ly/2JPNnGS) is used, the AUC will be printed using the challenge labels. We obtained an AUC ROC of 0.937 using the training challenge dataset for validation. THe test AUC was obtained in the challenge server with not-public labels.

