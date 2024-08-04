import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_excel(file_path)

def preprocess_data(df):
    # Dropping unnecessary columns
    df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    
    # Creating numerical and categorical columns
    num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 
                'TotalWorkingYears', 'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate', 
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']
    
    cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 
                'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'StockOptionLevel', 
                'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 
                'RelationshipSatisfaction']
    
    # Creating dummy variables
    to_get_dummies_for = ['BusinessTravel', 'Department', 'Education', 'EducationField', 
                          'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 
                          'JobRole', 'MaritalStatus']
    
    df = pd.get_dummies(data=df, columns=to_get_dummies_for, drop_first=True)
    
    # Mapping overtime and attrition
    dict_OverTime = {'Yes': 1, 'No': 0}
    dict_attrition = {'Yes': 1, 'No': 0}
    
    df['OverTime'] = df.OverTime.map(dict_OverTime)
    df['Attrition'] = df.Attrition.map(dict_attrition)
    
    # Separating target variable and other variables
    Y = df.Attrition
    X = df.drop(columns=['Attrition'])
    
    # Scaling the data
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Splitting the data
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
    
    return x_train, x_test, y_train, y_test
