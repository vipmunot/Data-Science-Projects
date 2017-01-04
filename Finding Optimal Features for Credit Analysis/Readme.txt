
We have self-implemented kNN and Naive Bayes code.
Besides that, we have used scikit libraries to perform analysis using SVM, kNN, Naive Bayes, Logistic Regression and Random Forest.
We have tried out various parameter settings to find out the best ones for each of the algorithm on each of the data set seperately.

The code for Naive Bayes is a separate file named NB_AML2.py, whereas the code for our kNN is present in all the dataset specific python files (German.py, CRX.py, Default.py)

final.py-Contains the best parameters settings by greedy approach for all three datasets.

Data set files:
1) german.data
2) crx.data
3) default of credit card clients.csv

Please see below on instructions on how to run these files:


1) Self Implemented Codes:
a) Naive Bayes
NB_AML2.py:
# NOTE: The program is python 2.7 compatible. Kindly run on compatible python version to avoid incorrect results/errors.
#
# This is the Naive Bayes Code:
# Please run this file with the following command:
# python NB_AML2.py data_file, delimiter, header_present_flag, label_index, fraction

# Example:
# 	python NB_AML2.py "default of credit card clients_no_headers.csv" "," False 15 0.1

# The parameters for all the 3 datasets are given below

# It takes 5 parameters as follows:
# 1) data_file = The data file name (with the full path if the file is not in the same folder as the python file)
# 	eg: "default of credit card clients_no_headers.csv"
#
# 2) delimiter = The delimiter used to separate the columns in the data (Can be "," or blankspace(" ") or anything else
# as long as its consistent through out the file)
# 	eg: ","
#
# 3) header_present_flag = True/False.Used to notify whether the data file contains the first row as column header or not
# 	eg: False
#
# 4) label_index = Need to give the index of the label column. (0,1,2....
#    NOTE: Index starts at 0 and NOT 1 !!!
# 		eg: 15
#
# 5) fraction: The fraction (should be always between 0 and 1) in which the data needs to be divided as training and test.
#    Eg: If 0.9 is given, then it randomly assigns 90% of the data as train and rest 10% as test.
# 		eg: 0.9

# Note: Please make sure you give data_file and header_present_flag in double quotes ("") and NOT single quotes ('')
#
# Parameters:
# 1) German Data Set (german.data)
# data_file = "german.data"
# delimiter = " " (Blank Space)
# header_present_flag = False
# Label Index = 20
# fraction = 0.9 (Can be changed to anything between 0 and 1)
#
# 2) CRX Data Set
# data_file = "crx.data"
# delimiter = "," (Comma)
# header_present_flag = False
# Label Index = 15
# fraction = 0.9 (Can be changed to anything between 0 and 1)
#
# 3) Default Data Set
# data_file = "default of credit card clients.csv"
# delimiter = "," (Comma)
# header_present_flag = True
# Label Index = 23
# fraction = 0.9 (Can be changed to anything between 0 and 1)




b) Self implemented kNN is present in the below python files itself

2) Scikit based implementation:
# Self implemented kNN is present in the below python files itself
# NOTE: The below python programs is python 3.5 compatible. Kindly run on compatible python version to avoid incorrect results/errors.
# Python version: 3.5
# Individual Dataset Analysis and Parameters Tuning:
# One python file per dataset
a) German.py
b) CRX.py
c) Default.py


Please run (directly run the files, no need for any parameters, as long as all the provided data files are in the same folder) the above 3 files separately. These files contains the PARAMETER TUNING by greedy approach for all three datasets.
Outputs for this is 10 Fold Cross Validation ROC graph and accuracy, and are stored in the below files respectively:
a) German.html
b) CRX.html
c) Default.html
The results of respective datasets is displayed as jupyter notebook in these files.
NOTE: For Default.py our kNN implementations takes a long time, due to large data.


Best Parameters settings and Confusion Matrix:
# Python version: 3.5
d) final.py
Please run final.py (having all data files in the same folder). 
This file contains the BEST PARAMETER SETTINGS, found while tuning in the above 3 files for all three datasets.
Outputs for this is 10 Fold Cross Validation ROC graph, accuracy and Confusion Matrix, and are stored in the below files respectively:

d) final.html = The results of final.py is displayed as jupyter notebook in this file.
