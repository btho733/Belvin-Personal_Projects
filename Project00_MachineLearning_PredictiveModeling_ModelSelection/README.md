# Model selection for predictive modeling tool (Early screening for oral cancer)

### For code : 
open the jupyter notebook Model_Selection.ipynb

### Quick Background :
This project explains the steps followed to select the optimal model for classifying oral cancer lesions based on colour images. A mass-screening tool was developed (in MATLAB/Python) based on this work. For more details, visit IIT Roorkee Masters thesis repository **(Belvin Thomas , "Identification and classification of oral cancer lesions in color images using SVM and ANN", 2013)**

The model selection with optimal parameters is an important step in the development of a predictive modelling tool which can efficiently handle the bias-variance trade-off. It ensures that the final model is capable of effectively handling the issues of underfitting and overfitting. **An ensemble of the selected models and associated parameters is suggested for optimum generalisation.** This will ensure unbiased prediction while dealing with in an unseen image in a real-world mass-screening scenario.

## The notebook file contains :

**1) Loading the cleaned data:** It contains texture features obtained from a repository of cancerous and non-cancerous images. Suitable features are selected from a set of texture features based on Gray level co-occurrance and Grey level run length. 

        For more details about the data and feature selection mechanism, visit the thesis cited above.

**2) Splitting of data:** Data os split into training-validation-test dataset at 60-20-20 ratio.

**3)Fitting a base model, cross validation and hyperparameter tuning based on following machine learning algorthms:**

         - Logistic Regression
         - Support Vector Machines (SVM)
         - Multi-Layer Perceptron (MLP)
         - Random forest classifier
         - Gradient Boosting classifier
         
Machine learning algorithm implementations from *scikit-learn library* is used to train the models. Hyperparameters are tuned using GridsearchCV

For the full dataset and more test data contact me belvinthomas@gmail.com
         
