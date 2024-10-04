import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import joblib
from imblearn.over_sampling import SMOTE

# the function to find out the column names based on the dtype
def column_types(loan_df):
    
    # separating numeric columns and object columns and float columns into three variables for label encoding and for limiting decimal points
    numerical_cols = loan_df.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    non_numerical_cols = loan_df.select_dtypes(include=['object', 'category']).columns.tolist()
    float_columns = loan_df.select_dtypes(include='float64').columns.tolist()

    # print(f"numerical_cols : {numerical_cols} \n\n non_numerical_cols : {non_numerical_cols} \n\n float_columns : {float_columns} \n\n")

    return numerical_cols, non_numerical_cols, float_columns

# All numerical columns are continuous columns and all non-numerical columns are categorical columns in the given dataset

# Function to limit the decmail points to 4 for the float columns
def decimal_four(loan_df):
    # limiting the decimal points to 4 for each float columns
    num, non_num, float_col = column_types(loan_df)
    loan_df[float_col] = loan_df[float_col].round(4)

    return loan_df

# Function to add extra feature columns for better prediction
def add_feature(loan_df):
    # Creating AGE_GROUP column to the dataset based on the age
    loan_df['Age_Group'] = pd.cut(loan_df['Age'], 
                                  bins=[17, 30, 50, 65, 100], 
                                  labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])

    # Creating Credit_Score_Group column from Credit_Score column
    loan_df['Credit_Score_Group'] = pd.cut(loan_df['Credit_Score'], 
                                           bins=[125, 580, 670, 740, 850], 
                                           labels=['Poor', 'Fair', 'Good', 'Excellent'])

    # Creating DTI_Bins column into the dataset
    loan_df['DTI_Group'] = pd.cut(loan_df['Debt_to_Income_Ratio'], 
                                  bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                  labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

    # Finding out EMI from Loan amount, Interest rate and Loan Duration in months column
    loan_df['Monthly_Loan_Payment'] = (loan_df['Loan_Amount'] * 
                                       (loan_df['Interest_Rate'] / 100)) / loan_df['Loan_Duration_Months']

    # Calculate Interest to Income Ratio from Interset rate, loan amount and income
    # how much of the applicant’s income is spent on paying interest alone for their loan. 
    # It helps in assessing whether the interest payments will be manageable based on the person's earnings.
    loan_df['Interest_to_Income_Ratio'] = ((loan_df['Interest_Rate'] * 
                                            loan_df['Loan_Amount']) / loan_df['Income']).round(4)

    # Calculate Loan to Income Ratio using the loan amount and income column
    # the proportion of the borrower’s income that the loan amount represents. 
    # tell us the applicant's borrowing capacity and whether the loan is reasonable compared to their earnings.
    loan_df['Loan_to_Income_Ratio'] = (loan_df['Loan_Amount'] / loan_df['Income']).round(4)

    # calling the decimal_four() function
    loan_df = decimal_four(loan_df)

    loan_df.to_csv("Added_Extra_columns.csv",index=False)
    
    return loan_df

# Functions to plot categorical columns
def cat_plot(col_name,loan_df):
    
    loan_df_copy = loan_df.copy()
    # Bar plot for a categorical column (e.g., 'Gender') vs target column 'Loan_Status'
    ax = sns.countplot(x=col_name, hue='Loan_Status', data=loan_df_copy)
    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue
        ax.annotate(f"{int(height)}",
                   (p.get_x() + p.get_width() / 2.,height),
                   ha = 'center', va = 'bottom',
                    fontsize = 10, color = 'black')
    
    plt.title(f'{col_name} Distribution by Loan Status')
    plt.show()

# Function to plot continuous columns
def con_plot(col_name,loan_df):

    loan_df_copy = loan_df.copy()
    # FacetGrid for a continuous column with respect to 'Loan_Status'
    g = sns.FacetGrid(loan_df, col='Loan_Status')
    
    # Plotting histograms
    g.map(sns.histplot, col_name, kde=True, bins=30)
    
    # Rename the facet column titles
    g.set_titles(col_template="{col_name}")  # Replace the default titles with the actual status labels
    g.axes[0, 0].set_title('Default')       # Set title for Loan_Status = 0 (Default)
    g.axes[0, 1].set_title('Non_Default')    # Set title for Loan_Status = 1 (Non_Default)
    
    g.add_legend()
    plt.show()

# Function to call both continuous plot and categorical plot together
def plotting(categorical_cols,continuous_cols,loan_df):
    for col in categorical_cols:
        cat_plot(col,loan_df)
    # Plotting for continuous value columns
    for con_cols in continuous_cols:
        con_plot(con_cols,loan_df)

# Function for label encoding
def label_encoding(loan_df):
    numerical_cols, non_numerical_cols, float_columns = column_types(loan_df)

    categorical_cols = non_numerical_cols.copy()

    continuous_cols = numerical_cols.copy()

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Dictionary to store the mappings for each column
    label_mappings = {}
    
    # Apply Label Encoding to each Non-Numerical Column and store mappings in the variable
    for col in non_numerical_cols:
        # Fit and transform the column
        loan_df[col] = label_encoder.fit_transform(loan_df[col].astype(str))
        
        # Capture the mapping of strings to numbers into lable_mappings variable
        mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
        label_mappings[col] = mapping

    # Display the mappings
    print("The following are the label encoded values")
    for col, mapping in label_mappings.items():
        print(f"Label Encoding for {col}: {mapping}")

    loan_df.to_csv("Label_encoded1.csv")

    return label_mappings, loan_df, continuous_cols, categorical_cols

# Function to print the label mappings
def encoded_labels(labels):
    # Display the mappings
    print("The following are the label encoded values")
    for col, mapping in labels.items():
        print(f"Label Encoding for {col}: {mapping}")

def preprocessing_df(loan_df,categorical_cols,continuous_cols):
    numerical_cols, non_numerical_cols, float_columns = column_types(loan_df)

    print(f"categorical_cols : {categorical_cols}")
    print()
    print(f"continuous_cols : {continuous_cols}")
    # OneHotEncoder for categorical values
    # Create the OneHotEncoder instance
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    # Fit and transform the categorical columns
    encoded_cols = encoder.fit_transform(loan_df[categorical_cols])
    # Creating a DataFrame with the encoded columns and get the feature names
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    # Drop the original categorical columns from loan_df
    loan_df = loan_df.drop(columns=categorical_cols)
    # Concatenate the original DataFrame with the new encoded DataFrame
    loan_df = pd.concat([loan_df, encoded_df], axis=1)

    # MinMaxScaler for continuous columns
    scaler = MinMaxScaler()
    loan_df[continuous_cols] = scaler.fit_transform(loan_df[continuous_cols])

    # Calling decimal four function
    loan_df = decimal_four(loan_df)

    # Save the normalized dataset into a new file
    loan_df.to_csv('Normalized.csv')

    return loan_df

def xy_split(loan_df):

    # Define features and target variable
    X = loan_df.drop('Loan_Status', axis=1)
    y = loan_df['Loan_Status']
    
    # Normalize feature
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return loan_df, X_train, X_test, y_train, y_test

def classimbalance_smote(loan_df,X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Print the class distribution after over-sampling
    print("Class distribution after SMOTE Oversampling:", y_train.value_counts())

    return loan_df,X_train, y_train

def feat_imp(loan_df,X_train, y_train):
    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for feature importances
    features = loan_df.drop('Loan_Status', axis=1).columns
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
    
    # Sort by importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    # Print feature importances
    print(feature_importances)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

    return loan_df, importances, features

# # Access column names where importance is below 0.076945
# low_importance_features = feature_importances[feature_importances['Importance'] < 0.058212]
# # Get the feature names
# low_importance_column_names = low_importance_features['Feature'].tolist()
# loan_df.drop(columns=low_importance_column_names, inplace=True)

# Model training
# Print model metrics
def print_model_metrics(model_name, model, y_true, y_pred, X_test):

    print(f"{model_name} Metrics:")
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, model.predict_proba(X_test)[:, 1]))
    print("\n")

# Function to plot the ROC Curve
def plot_roc_curve(model_name, model, X_test, y_test):
    # Get the predicted probabilities (for the positive class)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Dashed diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Updated model training function with ROC curve plotting
def model_training(X_train, y_train, X_test, y_test):

    # Dictionary to store model name and model object
    models = {}
    
    # LOGISTIC REGRESSION
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    # Make predictions
    y_pred_log_reg = log_reg.predict(X_test)
    # Store the model
    models["Logistic Regression"] = [log_reg,'log_reg']
    # Evaluate the model
    print_model_metrics("Logistic Regression", log_reg, y_test, y_pred_log_reg, X_test)
    plot_roc_curve("Logistic Regression", log_reg, X_test, y_test)

    # DECISION TREE
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    # Make predictions
    y_pred_tree = decision_tree.predict(X_test)
    # Store the model
    models["Decision Tree"] = [decision_tree,'decision_tree']
    # Evaluate the model
    print_model_metrics("Decision Tree", decision_tree, y_test, y_pred_tree, X_test)
    plot_roc_curve("Decision Tree", decision_tree, X_test, y_test)

    # RANDOM FOREST
    random_forest = RandomForestClassifier(random_state=42)
    random_forest.fit(X_train, y_train)
    # Make predictions
    y_pred_rf = random_forest.predict(X_test)
    # Store the model
    models["Random Forest"] = [random_forest,'random_forest']
    # Evaluate the model
    print_model_metrics("Random Forest", random_forest, y_test, y_pred_rf, X_test)
    plot_roc_curve("Random Forest", random_forest, X_test, y_test)

    # SVM (Support Vector Machines)
    svm = SVC(random_state=42, probability=True)  # Set probability=True for ROC curve
    svm.fit(X_train, y_train)
    # Make predictions
    y_pred_svm = svm.predict(X_test)
    # Store the model
    models["SVM"] = [svm,'svm']
    # Evaluate the model
    print_model_metrics("SVM", svm, y_test, y_pred_svm, X_test)
    plot_roc_curve("SVM", svm, X_test, y_test)

    return models

def hyp_tun(X_train, y_train, X_test, y_test):

    # Logistic regression model 
    def hyp_logre(X_train, y_train):
        
        # Define the parameter grid for Logistic Regression
        param_grid_log_reg = {
            'C': [0.001, 0.01, 0.1, 2, 10, 100],  # Regularization strength
            'solver': ['liblinear', 'saga', 'newton-cg', 'sag'],      # Solvers suitable for small datasets
            'max_iter': [90, 110, 200, 300]           # Number of iterations
        }
        
        # Initialize the model
        log_reg = LogisticRegression(random_state=42)
        
        # Initialize GridSearchCV
        grid_search_log_reg = GridSearchCV(estimator=log_reg, param_grid=param_grid_log_reg,
                                            scoring='accuracy', cv=5, n_jobs=-1)
        
        # Fit GridSearchCV
        grid_search_log_reg.fit(X_train, y_train)
        
        # Best parameters and score
        best_params_l = grid_search_log_reg.best_params_
        best_score_l = grid_search_log_reg.best_score_
        
        # Best parameters and best score
        print("Logistic Regresion")
        print("Best Parameters for Logistic Regression:", grid_search_log_reg.best_params_)
        print("Best Score for Logistic Regression:", grid_search_log_reg.best_score_)
        print()
        
        C_l = best_params_l['C']
        solver = best_params_l['solver']
        max_iter = best_params_l['max_iter']

        return C_l, solver, max_iter
    
    # Decision tree and random forest model
    def hyp_dtrf(X_train, y_train):
    
        # Define the parameter grid
        param_grid_rf = {
            'n_estimators': [90, 100, 150, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [4, 5, 7, 10],
            'min_samples_leaf': [2, 3, 4]
        }
        
        # Initialize RandomizedSearchCV
        rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, n_iter=10, cv=5, random_state=42, n_jobs=-1)
        
        # Fit the model
        rf_search.fit(X_train, y_train)
        
        # Best parameters and score
        best_params_r = rf_search.best_params_
        best_score_r = rf_search.best_score_
        
        # Best parameters and score
        print("Decision Tree and Random Forest")
        print("Best Parameters:", rf_search.best_params_)
        print("Best Score:", rf_search.best_score_)
        print()
        
        # Assign best parameters to individual variables
        n_estimators = best_params_r['n_estimators']
        max_depth = best_params_r['max_depth']
        min_samples_split = best_params_r['min_samples_split']
        min_samples_leaf = best_params_r['min_samples_leaf']
    
        return n_estimators, max_depth, min_samples_leaf, min_samples_split

    # SVM model
    def hyp_svm(X_train, y_train):
        
        # Define the parameter grid for SVM
        param_grid_svm = {
            'C': [0.001, 0.01, 0.1, 1, 2, 10, 100],  # Regularization parameter
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
            'gamma': ['scale', 'auto', 0.1, 1, 5, 10]  # Kernel coefficient
        }
        
        # Initialize the model
        svm = SVC(random_state=42)
        
        # Initialize GridSearchCV
        grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm,
                                        scoring='accuracy', cv=5, n_jobs=-1)
        
        # Fit GridSearchCV
        grid_search_svm.fit(X_train, y_train)
        
        # Best parameters and score
        best_params_s = grid_search_svm.best_params_
        best_score_s = grid_search_svm.best_score_
        
        # Best parameters and best score
        print("Support Vector Machine")
        print("Best Parameters for SVM:", grid_search_svm.best_params_)
        print("Best Score for SVM:", grid_search_svm.best_score_)
        print()
        
        C_s = best_params_s['C']
        kernel = best_params_s['kernel']
        gamma = best_params_s['gamma']

        return C_s, kernel, gamma


    # Call both hyperparameter tuning functions
    rf_results = hyp_dtrf(X_train, y_train)
    log_reg_results = hyp_logre(X_train, y_train)
    svm_results = hyp_svm(X_train, y_train)

    # Combine results from both models
    return (*rf_results, *log_reg_results, *svm_results)

# Function to train models and plot ROC curves
def hy_model_training(X_train, y_train, X_test, y_test, C_l, max_iter, solver, 
                      min_samples_split, min_samples_leaf, max_depth, 
                      n_estimators, C_s, gamma, kernel):

    # Dictionary to store model name and model object
    hyp_models = {}

    # LOGISTIC REGRESSION
    hyp_log_reg = LogisticRegression(C=C_l, max_iter=max_iter, solver=solver, random_state=42)
    hyp_log_reg.fit(X_train, y_train)
    y_pred_log_reg = hyp_log_reg.predict(X_test)
    # Store the model
    hyp_models["Logistic Regression"] = [hyp_log_reg,'hyp_log_reg']
    # Calling print_model_metrics function
    # hy_log_reg = print_model_metrics("Logistic Regression", hyp_log_reg, y_test, y_pred_log_reg, X_test)
    print_model_metrics("Logistic Regression", hyp_log_reg, y_test, y_pred_log_reg, X_test)
    # Call plot_roc_curve function
    plot_roc_curve("Logistic Regression", hyp_log_reg, X_test, y_test)

    # DECISION TREE
    hyp_decision_tree = DecisionTreeClassifier(
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=42
    )
    hyp_decision_tree.fit(X_train, y_train)
    y_pred_tree = hyp_decision_tree.predict(X_test)
    # Store the model
    hyp_models["Decision Tree"] = [hyp_decision_tree,'hyp_decision_tree']
    # hyp_decision_tr = print_model_metrics("Decision Tree", hyp_decision_tree, y_test, y_pred_tree, X_test)
    print_model_metrics("Decision Tree", hyp_decision_tree, y_test, y_pred_tree, X_test)
    plot_roc_curve("Decision Tree", hyp_decision_tree, X_test, y_test)

    # RANDOM FOREST
    hyp_random_forest = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        random_state=42
    )
    hyp_random_forest.fit(X_train, y_train)
    y_pred_rf = hyp_random_forest.predict(X_test)
    # Store the model
    hyp_models["Random Forest"] = [hyp_random_forest,'hyp_random_forest']
    # hyp_random_fr = print_model_metrics("Random Forest", hyp_random_forest, y_test, y_pred_rf, X_test)
    print_model_metrics("Random Forest", hyp_random_forest, y_test, y_pred_rf, X_test)
    plot_roc_curve("Random Forest", hyp_random_forest, X_test, y_test)
 
    # SVM (with probability=True)
    hyp_svm = SVC(C=C_s, gamma=gamma, kernel=kernel, probability=True, random_state=42)
    hyp_svm.fit(X_train, y_train)
    y_pred_svm = hyp_svm.predict(X_test)
    # Store the model
    hyp_models["SVM"] = [hyp_svm,'hyp_svm']
    # hyp_svm_mdl = print_model_metrics("SVM", hyp_svm, y_test, y_pred_svm, X_test)
    print_model_metrics("SVM", hyp_svm, y_test, y_pred_svm, X_test)
    plot_roc_curve("SVM", hyp_svm, X_test, y_test)
    
    # return hy_log_reg, hyp_decision_tr, hyp_random_fr, hyp_svm_mdl
    return hyp_models

# Function to save the preferred model
def save_model(model):
    # Save the SVM model
    joblib.dump(model, 'loan_df_model.pkl')

# this is the main function which loads the dataseta and calls all other functions
def main():

    # Load data into python
    loan_df = pd.read_csv('loan_default_prediction_project.csv')
    
    # Fill the NaN columns in the Employment_Status column with the mean of the median value of Employed and Unemployed income
    emp_med = loan_df[loan_df['Employment_Status'] == 'Employed']['Income'].median()
    unemp_med = loan_df[loan_df['Employment_Status'] == 'Unemployed']['Income'].median()
    val = (emp_med + unemp_med) / 2

    # Update null Employment_Status column
    loan_df.loc[(loan_df['Employment_Status'].isnull()) & (loan_df['Income'] >= val), 'Employment_Status'] = 'Employed'
    loan_df.loc[(loan_df['Employment_Status'].isnull()) & (loan_df['Income'] < val), 'Employment_Status'] = 'Unemployed'

    # Fill NaN Gender column cells with Female and Male values alternately
    gen_vals = ['Female', 'Male']
    num_nulls = loan_df['Gender'].isnull().sum()
    gen_vals_l = []

    # Creating a list with Female and Male values with items equal to the number of null cells
    for i in range(num_nulls):
        gen_vals_l.append(gen_vals[i % 2])

    # Fill the null entries with the selected alternating values
    loan_df.loc[loan_df['Gender'].isnull(), 'Gender'] = gen_vals_l

    # calling the decimal_four() function
    loan_df = decimal_four(loan_df)

    # Saving the cleaned dataset into a new file for easy access
    loan_df.to_csv('Null_Values_Handled.csv',index=False)

    # callig the add_feature function to add extra features into the dataframe
    loan_df = add_feature(loan_df)

    # Capturing Categorical, continuous column names
    numerical_cols, non_numerical_cols, float_columns = column_types(loan_df)
    # creating categorical and continuous value columns for plotting
    categorical_cols = non_numerical_cols.copy()
    continuous_cols = numerical_cols.copy()
    # removing target column from categorical columns
    categorical_cols.remove('Loan_Status')
    
    plotting(categorical_cols,continuous_cols,loan_df)

    # calling function for label encoding
    label_mappings, loan_df, continuous_cols, categorical_cols = label_encoding(loan_df)


    categorical_cols.remove("Loan_Status")
    # print(f"categorical_cols : {categorical_cols} \n\n continuous_cols : {continuous_cols} \n\n")
    
    loan_df = preprocessing_df(loan_df,categorical_cols,continuous_cols)

    loan_df, X_train, X_test, y_train, y_test = xy_split(loan_df)

    loan_df,X_train, y_train = classimbalance_smote(loan_df,X_train, y_train)

    loan_df, importances, features = feat_imp(loan_df,X_train, y_train)

    def_models = model_training(X_train, y_train, X_test, y_test)

    n_estimators, max_depth, min_samples_leaf, min_samples_split, C_l, solver, max_iter, C_s, kernel, gamma = hyp_tun(X_train, y_train, X_test, y_test)

    hyp_models = hy_model_training(X_train, y_train, X_test, y_test, C_l, max_iter, solver, 
                      min_samples_split, min_samples_leaf, max_depth, 
                      n_estimators, C_s, gamma, kernel)

    
    # Combine def_models and hyp_models into one dictionary called "models"
    models = {**def_models, **hyp_models}

    # print("Default Models")
    # # Iterate over the dictionary to unpack model name and model
    # for model_name, model in models.items():
    #     print(f"Model_name : {model_name} - Model is : {model}")

    # # Iterate over the dictionary to unpack model name and model
    # for model_name, model in hyp_models.items():
    #     print(f"Model_name : {model_name} - Model is : {model}")
    #     print()
        
    model_l = ['log_reg','decision_tree','random_forest','svm','hyp_log_reg','hyp_decision_tree','hyp_random_forest','hyp_svm']

    print("Select any of the model which you need to save from the list")
    for i in range(len(model_l)):
        print()
        print(f"{i} - {model_l[i]}")

    sel_model = int(input("Enter the model number to save the model (0 to 7) press any other key to exit without saving"))

    if sel_model in range(0,8):
        model_name = model_l[sel_model]
        save_model(model_name)
        print(f"Model : {model_name} saved")
    else:
        pass

    # Return the updated loan_df
    return loan_df

# Call the main function to run the entire pipeline
if __name__ == "__main__":
    main()