from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor

# This python file returns a dF with all the metrics
def MetricsResults (y_train, y_pred_train,y_test,y_pred_test):
    
    import pandas as pd
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    
    def Metrics_df(R_train,R_test):
        met = {'metrics':['R2','MSE','RMSE','MAE'],'Train':R_train,'Test':R_test}
        return pd.DataFrame(met)
    
    def Metrics(y_true, y_pred):
        R2 = round(r2_score(y_true, y_pred),2)
        MSE = round(mean_squared_error(y_true, y_pred, squared=True),2)
        RMSE = round(mean_squared_error(y_true, y_pred, squared=False),2)
        MAE = round(mean_absolute_error(y_true, y_pred),2)
        return [R2,MSE,RMSE,MAE]

    return Metrics_df(Metrics(y_train, y_pred_train),Metrics(y_test, y_pred_test))


# This function is used to check the R2 relation with the train and test set for the KNN model
def r2_train_test(X_normalized_train_age,X_normalized_test_age,y_train_age, y_test_age,k_max = 3,weights = 'uniform'):
    
    r2_train = []
    r2_test = []
    n_neighbors = list(range(1,k_max))
    for k in n_neighbors:
        knn = KNeighborsRegressor(n_neighbors=k, weights = weights) 
        knn.fit(X_normalized_train_age,y_train_age)

        pred = knn.predict(X_normalized_train_age) 
        r2_train.append(r2_score(y_train_age, pred))

        pred = knn.predict(X_normalized_test_age) 
        r2_test.append(r2_score(y_test_age, pred))

    plt.scatter(n_neighbors,r2_train)
    plt.scatter(n_neighbors,r2_test)
    plt.xlabel("K number")
    plt.ylabel("R2")
    plt.legend(['train','test'])
    plt.show()





def Dependance_test(df,col1,col2,chi_val):
    data_crosstab = pd.crosstab(df[col1], df[col2], margins = False)
    if (st.chi2_contingency(data_crosstab)[1] < chi_val):
#         print(col1, 'and', col2 , 'are DEPENDENT ', st.chi2_contingency(data_crosstab)[1])
        return np.round(st.chi2_contingency(data_crosstab)[1],2)
    else:
#         print(col1, 'and', col2 , 'are INDEPENDENT ', st.chi2_contingency(data_crosstab)[1])
        return np.round(st.chi2_contingency(data_crosstab)[1],2)

def Dependance_matrix(df):
    rows = []
    cols = []
    for col1 in df.columns:
        for col2 in df.columns:
            rows.append(Dependance_test(df,col1,col2,0.05))
        cols.append(rows)
        rows = []
    return pd.DataFrame(cols, columns = df.columns,index = df.columns)