# PetFinderTask
Pet Finder analysis using the XGBoost model


## Objective

## Task 1
* Using the PetFinder dataset, train an ML model using XGB to predict whether a pet will be adopted or not `Adopted` is the target feature. 
* You will need to use the validation set to assess early stopping. You won't need to hypertune any parameter, the default parameters will be sufficient, with the exception of the number of trees which gets tuned by the early stopping mechanism.
* The script needs to log to the user the performances of the model in the test set in terms of F1 Score, Accuracy, Recall.
* Save the model.

## Task 2
* Load the model and predict on all rows of the PetFinder CSV file.
* Save the results.
* Add a unit test to the prediction function.

The output needs to follow the following format:
```
Type,Age,Breed1,Gender,Color1,Color2,MaturitySize,FurLength,Vaccinated,Sterilized,Health,Fee,PhotoAmt,Adopted,Adopted_prediction
Cat,3,Tabby,Male,Black,White,Small,Short,No,No,Healthy,100,1,Yes, No
```

## Approach
* Ran analysis and model as Jupyter notebook files
* Conda environment saved as environment.yml for reproducibility
* Added the GCSFS library to support Google Cloud Storage - see https://gcsfs.readthedocs.io/en/latest/
* Performed EDA on dataset
  * Numerical features
  * Categorical features
  * Checked for missing values (none found)
* XGBoost has experimental support for categorical features now but decided against using it
* Converted categorical features into numeric using the one-hot encoding technique
* "Breed1" feature has high cardinality (166). Hot-encoding this column would result in the curse of dimensionality so wrote a function to group together all of the smaller categories under "Other"
* Functions are stored in a separate Python file as they are shared between the two notebooks
* A baseline model was included for comparison


## Further work / next steps
* Consider modelling cat and dog adoption separately
* The class examples are unbalanced (3:1 for adopted: not adopted). May consider stratified sampling to see if the model can be improved.
* Consider using Weight of Evidence and Information Value (both used in Credit Scoring, https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
