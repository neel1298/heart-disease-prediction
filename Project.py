#Heart Disease Prediction by Logistic Regression

#Het Patel - 8680821
#Neel Patel - 8682068



import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
#import matplotlib.mlab as mlab
from statsmodels.tools import add_constant as add_constant
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
from sklearn.metrics import roc_curve



def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    plt.show()

def back_feature_elem (data_frame,dep_var,col_list):
    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)


#Read csv file into a Pandas Dataframe
framingham_df = pd.read_csv('framingham.csv')
#Remove education column
framingham_df.drop(['education'],axis=1,inplace=True)


########################################### Data Preparation ###########################################
print("################################### Data Preparation ###################################")

#Check datatypes to make sure they are suitable (numeric) to perform regression on
print("\nCheck Data Types:")
print(framingham_df.dtypes)


#Check for duplicate values
print("\nCheck Duplicate Values:")
dup_values_df = framingham_df[framingham_df.duplicated()]
if(len(dup_values_df) < 1):
    print("There are no duplicate values")
else:
    print(dup_values_df)


#Count number of rows with null values and remove the null values if they do not take up a majority part of the dataset
print("\nCheck for rows with missing values:")
null_row_count = 0
for i in framingham_df.isnull().sum(axis=1):
    if i > 0:
        null_row_count = null_row_count + 1

print('Total number of rows with missing values:', null_row_count)
print('Since it takes up only',round((null_row_count/len(framingham_df.index))*100), 'percent of the entire dataset, the rows with missing values will be removed.')

#Drop columns with NA
framingham_df.dropna(axis=0,inplace=True)


########################################### Descriptive Data Analysis ###########################################
print("\n################################### Descriptive Data Analysis ###################################")

print("\nNumerical summary of data:")
print(framingham_df.describe())

print("\nGraphical summary of data:")
draw_histograms(framingham_df,framingham_df.columns,6,3)

#histogram for categorical variable TenYearCHD
sn.countplot(x='TenYearCHD',data=framingham_df)

#Correlation heat table 
sn.heatmap(framingham_df.corr())
plt.show()


########################################### Logistic Regression ###########################################
print("\n################################### Logistic regression ###################################")

#Add column of 1's to the data
framingham_df_constant = add_constant(framingham_df)

st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=framingham_df_constant.columns[:-1]
model=sm.Logit(framingham_df.TenYearCHD,framingham_df_constant[cols])
result=model.fit()
print("\nBaseline Model:")
print(result.summary())

#Backward selection model (elimination of non significant variables by performing regression repeatedly)
print("\n Backward Selection Model:")
result=back_feature_elem(framingham_df_constant,framingham_df.TenYearCHD,cols)
print(result.summary())

########################################### Interpreting the results: Odds Ratio, Confidence Intervals and Pvalues ###########################################
print("\n### Interpreting the results: Odds Ratio, Confidence Intervals and Pvalues ###")
params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))


########################################### Computing accuracy ###########################################
print("\n############### Computing Accuracy ##################")

#Splitting data to train and test split
new_features=framingham_df[['age','male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,test_size=.1,random_state=0)

#Apply model to train data
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

#Compute Accuracy
print("Accuracy of Model:",(sklearn.metrics.accuracy_score(y_test,y_pred))*100,"%")


########################################### Model Evaluation by Confusion Matrix ###########################################
print("\n################## Model Evaluation by Confusion Matrix ##################")
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")



TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)

print('\nThe accuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)



###### FINAL PREDICTION ######
print("\n#################### FINAL PREDICTION #######################\n")
y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
print(y_pred_prob_df)


#Threshold adjustment to increase sensitivity
print("\nThreshold adjustment:")
for i in range(1,5):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    
    
########################################### ROC curve and AUC ###########################################
print("\n################################### ROC Curve and AUC ###################################")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Model')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

#AUC
print("AUC:",sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1]))
