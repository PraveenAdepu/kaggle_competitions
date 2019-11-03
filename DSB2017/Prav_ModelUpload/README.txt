Author : Praveen Adepu

Competition : DSB2017

Note : These models has very basic implementation and intended to experiment with deep learning rather to produce production level accurate predictions

Scripts in order to reproduce results:

1. 0000.Setup.R
		To setup all working directories and required packages
2. 0001.Prav_CV_5folds.R
		This will partition the training data into 5 folds
		Output : CVindices csv file
External Data : resnet50 from http://data.dmlc.ml/mxnet/models/imagenet/resnet/50-layers/
3. 1000.FeatureExtraction_00.py
		This is first step to extract simple features and requires adjusting working directory paths
		Output: npy files
4. 1001.FeatureExtraction_01.py
		This is secondary method to extract features and requires adjusting working directory paths
5. 2000.Prav_FE01_xgb01.py
6. 2001.Prav_FE01_nn01.py
7. 2002.Prav_FE00_nn01.py
8. 2003.Prav_FE00_xgb01.py

9. 3000.Prav_Ensemble to get submitted files

		