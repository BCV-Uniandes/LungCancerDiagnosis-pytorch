# Automated Diagnosis of Lung Cancer with 3D Convolutional Neural Networks

Pytorch code for the [Automated Prediction of Lung Cancer with 3D Convolutional Neural Networks](https://biomedicalcomputervision.uniandes.edu.co/index.php/research?id=32). Our cancer predictor obtained a ROC AUC of 0.913 and was ranked 1st place at the [ISBI 2018 Lung Nodule Malignancy Prediction challenge](https://bit.ly/2JPNnGS).

### Prerequisites

This code was implemented in Python 2.7.*, using PyTorch, Numpy, pandas, sklearn, scipy, skimage and dicom.

### Getting started

To run the code save the folder of each patient with the dicom files (of the ISBI 2018 Lung challenge) in the folder ./data/ISBI-deid-TRAIN/ and run ./test_ISBI.py

To run the code with a different ling CT scan, save the folder with the dicom files in the folder ./data/ISBI-deid-TRAIN/ and run ./test.py. For scans different from the ISBI 2018 Lung challenge dataset, the program will output the score after the predictor (without the mask post-processing).

Download the trained models from this [link](https://www.dropbox.com/s/to7pmlajtr0tyos/models.zip?dl=0). Detector model was trained with the [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) and the predictor with the [Kaggle DSB2017 dataset](https://www.kaggle.com/c/data-science-bowl-2017).

If the dataset from the [ISBI 2018 Lung Nodule Malignancy Prediction challenge](https://bit.ly/2JPNnGS) is used, the AUC will be printed using the challenge labels. We obtained an AUC ROC of 0.937 using the training challenge dataset for validation. The test AUC (91.3) was obtained in the challenge server with not-public labels.

In folder ./data/sorted_slices_jpgs/ the program will save images of the axial, sagittal and coronal planes of the 30 detected nodules with highest score of each patient.


