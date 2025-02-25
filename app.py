import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import precision_recall_curve
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label =LabelEncoder()
        for col in data.columns:
            data[col]=label.fit_transform(data[col])
        return data
    
    @st.cache(persist=True)
    def split(df):
        y=data['class']
        x=data.drop('class',axis=1)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train,x_test,y_train,y_test
    
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, model.predict(x_test))  # Compute Confusion Matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)  # Show in Streamlit

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            y_scores = model.predict_proba(x_test)[:, 1]  # Get probability scores
            fpr, tpr, _ = roc_curve(y_test, y_scores)  # Compute ROC curve
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc='lower right')
            st.pyplot(fig)  # Show in Streamlit

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            y_scores = model.predict_proba(x_test)[:, 1]  # Get probability scores
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label="Precision-Recall Curve")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            st.pyplot(fig)  # Show in Streamlit
    data=load_data()
    x_train,x_test,y_train,y_test=split(data)
    class_names=['edible','poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier=st.sidebar.selectbox("Classifier",("Logistic Regression","Random Forest","Support Vector Machine"))
    print(classifier)
    if classifier =='Support Vector Machine':
        c=st.sidebar.slider("C",0.01,10.0)
        kernel=st.sidebar.selectbox("Kernel",("linear","rbf","poly"))
        gamma=st.sidebar.selectbox("Gamma",("scale","auto"))
        metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("Classify"):
            st.subheader("Support Vector Machine Results")
            model=SVC(C=c,kernel=kernel,gamma=gamma, probability=True)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ",accuracy)
            st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier =='Logistic Regression':
        c=st.sidebar.slider("C",0.01,10.0)
        max_iter=st.sidebar.slider("Max Iterations",100,500)
        metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("Classify"):
            st.subheader("Logistic Regression Results")
            model=LogisticRegression(C=c,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ",accuracy)
            st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names))
            plot_metrics(metrics, model, x_test, y_test, class_names)
    if classifier =='Random Forest':
        n_estimators=st.sidebar.slider("Number of Estimators",100,500)
        max_depth=st.sidebar.slider("Max Depth",1,20)
        metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))
        if st.sidebar.button("Classify"):
            st.subheader("Random Forest Results")
            model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("Accuracy: ",accuracy)
            st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names))
            st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names))
            plot_metrics(metrics, model, x_test, y_test, class_names)
    if st.sidebar.checkbox("Show raw data",False):
        st.subheader("Mushroom Data Set")
        st.write(data)
    


if __name__=='__main__':
    main()