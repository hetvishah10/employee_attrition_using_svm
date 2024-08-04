import pandas as pd
from src.data.make_dataset import load_data, preprocess_data
from src.visualization.visualize import plot_histograms, plot_categorical_analysis, plot_correlation_matrix, plot_scatter
from src.models.train_model import fit_logistic_regression, fit_svm, metrics_score
from src.models.predict_model import predict

def main():
    file_path = 'C:/Users/Owner/Desktop/Hetvi Shah/Assignment_Data_Science/employee_attrition_using_svm/data/raw/HR_Employee_Attrition.xlsx'
    df = load_data(file_path)
    
    # Display the dataset info and sample
    print(df.sample(5))
    print(df.info())
    print(df.nunique())
    
    # Data Preprocessing
    x_train, x_test, y_train, y_test = preprocess_data(df)
    
    # Plotting
    num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike',
                'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate',
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']
    
    cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField',
                'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'StockOptionLevel', 
                'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 
                'RelationshipSatisfaction']
    
    plot_histograms(df, num_cols)
    plot_categorical_analysis(df, cat_cols)
    plot_correlation_matrix(df, num_cols)
    
    # Fit Logistic Regression
    lg = fit_logistic_regression(x_train, y_train)
    y_pred_train = predict(lg, x_train)
    metrics_score(y_train, y_pred_train)
    
    # Fit SVM with different kernels
    kernels = ['linear', 'rbf', 'poly']
    for kernel in kernels:
        print(f"Training SVM with {kernel} kernel")
        svm = fit_svm(x_train, y_train, kernel=kernel)
        y_pred_train_svm = predict(svm, x_train)
        metrics_score(y_train, y_pred_train_svm)
        y_pred_test_svm = predict(svm, x_test)
        metrics_score(y_test, y_pred_test_svm)

if __name__ == '__main__':
    main()