import seaborn as sns
import matplotlib.pyplot as plt
def results_drawer(y_pred_train,y_train,y_pred_test,y_test):
    pl = y_train.copy()
    pl['pred'] = y_pred_train
    pl2 = y_test.copy()
    pl2['pred'] = y_pred_test
    sns.set(rc={'figure.figsize':(15,9)})
    fig, ax = plt.subplots(2,2)
    sns.scatterplot(data=pl, x = 'pred', y = 'total_claim_amount', ax=ax[0,0])
    sns.scatterplot(data=pl2, x = 'pred', y = 'total_claim_amount', ax=ax[0,1])
    sns.histplot(data=pl.pred-pl.total_claim_amount, ax=ax[1,0])
    sns.histplot(data=pl2.pred-pl2.total_claim_amount, ax=ax[1,1])