from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
import shap
import pickle

main = Tk()
main.title("Detection of DGA Botnets using Explainable AI")
main.geometry("1300x1200")

global filename
global dataset
global X, Y
global X_train, X_test, y_train, y_test
labels = ['Normal', 'DGA Botnet']
global accuracy, precision, recall, fscore, scaler, columns, rf_cls

#fucntion to upload dataset
def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename) #read dataset from uploaded file
    text.insert(END,"Dataset Values\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    
    label = dataset.groupby('dga').size()
    label.plot(kind="bar")
    plt.xlabel('DGA Type')
    plt.ylabel('Count')
    plt.title("Dataset Detail 0 (Normal) & 1 (DGA Botnet)")
    plt.show()

def preprocess():
    text.delete('1.0', END)
    global dataset, scaler, columns
    global X_train, X_test, y_train, y_test, X, Y
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    columns = dataset.columns
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else:
        dataset = dataset.values
        X = dataset[:,0:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices) #shuffle dataset
        X = X[indices]
        Y = Y[indices]
        X = X[0:20000]
        Y = Y[0:20000]
        np.save("model/X", X)
        np.save("model/Y", Y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset After Features Processing & Normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset : "+str(X.shape[1])+"\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")
    

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()
    CR = classification_report(y_test, predict,target_names=labels)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")
    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runRandomForest():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, rf_cls
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train, y_train)
    predict = rf_cls.predict(X_test)
    calculateMetrics("Random Forest", predict, y_test)

def runLogisticRegression():
    lr_cls = LogisticRegression()
    lr_cls.fit(X_train, y_train)
    predict = lr_cls.predict(X_test)
    calculateMetrics("Logistic Regression", predict, y_test)

def runNaiveBayes():
    nb_cls = GaussianNB()
    nb_cls.fit(X_train, y_train)
    predict = nb_cls.predict(X_test)
    calculateMetrics("Naive Bayes", predict, y_test)

def runExtraTree():
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(X_train, y_train)
    predict = dt_cls.predict(X_test)
    calculateMetrics("Extra Tree", predict, y_test)    

def runEnsemble():
    bc_cls = BaggingClassifier()
    bc_cls.fit(X_train, y_train)
    predict = bc_cls.predict(X_test)
    calculateMetrics("Bagging Classifier", predict, y_test) 

def runXGBoost():
    global rf_cls
    X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)
    xg_cls = XGBClassifier()
    xg_cls.fit(X_train, y_train)
    rf_cls = xg_cls
    predict = xg_cls.predict(X_test)
    calculateMetrics("Proposed XGBoost Classifier", predict, y_test)     

def runXAI():
    global X_test, rf_cls, columns
    explainer = shap.TreeExplainer(rf_cls)
    shap_values = explainer.shap_values(X_test[0:200])
    shap.summary_plot(shap_values, features=X_test[0:200], feature_names=columns)

def predict():
    global scaler, rf_cls, dataset, filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = scaler.transform(dataset)
    predict = rf_cls.predict(X) 
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"Test Data : "+str(dataset[i])+" ====> NORMAL\n\n")
        else:
            text.insert(END,"Test Data : "+str(dataset[i])+" ====> DGA BOTNET DETECTED\n\n")            

font = ('times',18, 'bold')
title = Label(main, text='A Collaborative Detection Against BotNet Attacks using Explainable AI')
title.config(bg='#800020', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Botnet DGA Dataset", command=uploadDataset)
uploadButton.place(x=500,y=550)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=800,y=550)
processButton.config(font=ff)

rfButton = Button(main, text="Run Random Forest", command=runRandomForest)
rfButton.place(x=280,y=600)
rfButton.config(font=ff)

lrButton = Button(main, text="Run Logistic Regression", command=runLogisticRegression)
lrButton.place(x=450,y=600)
lrButton.config(font=ff)

nbButton = Button(main, text="Run Naive Bayes", command=runNaiveBayes)
nbButton.place(x=670,y=600)
nbButton.config(font=ff)

etButton = Button(main, text="Run Extra Tree", command=runExtraTree)
etButton.place(x=850,y=600)
etButton.config(font=ff)

eaButton = Button(main, text="Run Bagging Classifier", command=runEnsemble)
eaButton.place(x=1000,y=600)
eaButton.config(font=ff)

extensionButton = Button(main, text="Run Proposed XGBoost", command=runXGBoost)
extensionButton.place(x=420,y=650)
extensionButton.config(font=ff)

graphButton = Button(main, text="Shapely XAI Analysis", command=runXAI)
graphButton.place(x=650,y=650)
graphButton.config(font=ff)

predictButton = Button(main, text="Predict DGA Botnet from Test Data", command=predict)
predictButton.place(x=870,y=650)
predictButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=185)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='#D8C3A5')
main.mainloop()
