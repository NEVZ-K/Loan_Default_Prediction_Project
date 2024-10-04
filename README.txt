 Loan Default Prediction Project README

Overview

This project builds a predictive model to assess the risk of loan default for loan applicants. The code is structured into multiple functions that perform tasks such as data cleaning, feature engineering, model training, hyperparameter tuning, and model evaluation. The final model can be selected and saved based on user input.

Prerequisites

Before running the code, ensure that the following Python libraries are installed:
•	pandas
•	numpy
•	scikit-learn
•	imbalanced-learn
•	matplotlib
•	joblib

You can install these libraries using pip:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib joblib

Instructions to Run the Code

Step 1: Load the Dataset

Make sure that the dataset file (`loan_default_prediction_project.csv`) is in the same directory as your script. The code loads this dataset in the main function.

Step 2: Run All Functions

Before calling the `main()` function, you need to ensure that all the required functions are run. These functions include:

1. Data Handling Functions:
•	`column_types()`
•	`decimal_four()`
•	`add_feature()`

2. Plotting Functions:
•	`cat_plot()`
•	`con_plot()`
•	`plotting()`

3. Preprocessing Functions:
•	`label_encoding()`
•	`preprocessing_df()`

4. Feature Engineering and Data Splitting:
•	`xy_split()`
•	`classimbalance_smote()`
•	`feat_imp()`

5. Model Functions:
•	`print_model_metrics()`
•	`plot_roc_curve()`
•	`model_training()`
•	`hyp_tun()`
•	`hy_model_training()`
•	`save_model()`

Step 3: Call the Main Function

After running all the functions, call the `main()` function to execute the entire pipeline. This will:
1.	Load the dataset.
2.	Perform data cleaning and feature engineering.
3.	Train default models (Logistic Regression, Decision Tree, Random Forest, SVM).
4.	Perform hyperparameter tuning on these models.
5.	Allow you to select and save the trained model based on user input.


To run the main function, execute the following line at the end of your script:
python
if __name__ == "__main__":
    main()


Output Files

The following CSV files will be generated during execution:
•	`Null_Values_Handled.csv`: Dataset with handled missing values.
•	`Added_Extra_columns.csv`: Dataset with added feature columns.
•	`Label_encoded1.csv`: Dataset after label encoding.
•	`Normalized.csv`: Dataset after normalization and one-hot encoding.

The trained model can be saved as `loan_df_model.pkl` based on your selection.

Final Note

This pipeline covers everything from data preprocessing to model training and evaluation, ensuring flexibility for further modifications and usage for different datasets.
