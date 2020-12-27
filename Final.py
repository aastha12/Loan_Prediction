#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import streamlit.components.v1 as components


# In[3]:


import joblib


# In[4]:


import sweetviz as sv


# In[5]:


from pycaret.classification import *


# In[6]:


tuned_rf=joblib.load('RF.pkl')
tuned_cat=joblib.load('Cat.pkl')
tuned_et=joblib.load('ET.pkl')


# In[7]:


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('CPU') 


# In[8]:


from tensorflow.keras.models import load_model
from keras.initializers import he_normal

classifier_ann = load_model('my_model.hdf5',custom_objects={'HeNormal': he_normal()},compile=False)


# In[9]:


def run():

    #from PIL import Image
    #image = Image.open('logo.png')
    #image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "What would you like to do?",
    ("Online Prediction","Batch Prediction"))

    st.sidebar.info('This app is created to predict if the applicant should be granted a loan or not.')
    
    #st.sidebar.image(image_hospital)

    st.title("Loan Prediction App")
    
    if add_selectbox == 'Online Prediction':
        
        gender = st.selectbox('Gender',['Female','Male'])
        married = st.selectbox('Married',['No','Yes'])
        depend = st.selectbox('Dependents',['0','1','2','3+'])
        edu = st.selectbox('Education',['Graduate','Not Graduate'])
        self = st.selectbox('Self Employed',['No','Yes'])
        app_inc = st.number_input ('Applicant Income')
        co_inc = st.number_input ('Coapplicant Income')
        amt = st.number_input ('Loan Amount')
        term = st.number_input ('Loan Amount Term')
        credit = st.selectbox('Credit History',['0','1'])
        prop_are = st.selectbox('Property Area',['Rural','Semiurban','Urban'])

        output=""

        test_df = pd.DataFrame()
        test_df['Gender']= [gender]
        test_df['Married']=[married]
        test_df['Dependents']=[depend] 
        test_df['Education']=[edu]
        test_df['Self_Employed']=[self] 
        test_df['ApplicantIncome']=[app_inc] 
        test_df['CoapplicantIncome']=[co_inc] 
        test_df['LoanAmount']=[amt] 
        test_df['Loan_Amount_Term']=[term] 
        test_df['Credit_History']=[credit] 
        test_df['Property_Area']=[prop_are]      
        

        if st.button("Predict"):
            RF_pred=predict_model(tuned_rf,data=test_df)['Label']
            Cat_pred=predict_model(tuned_cat,data=test_df)['Label']
            Et_pred=predict_model(tuned_et,data=test_df)['Label']


            NN_input=pd.DataFrame(RF_pred)
            NN_input.columns=['RF']
            NN_input['Cat']=Cat_pred
            NN_input['ET']=Et_pred

            predictions=classifier_ann.predict_classes(NN_input)

            output = predictions
            
            if(output==0):
                text="Rejected"
                st.error(text)
                
            elif(output==1):
                text="Approved"
                st.success(text)
 
    
    if add_selectbox == 'Batch Prediction':

        file_upload = st.file_uploader("Upload excel file for predictions", type=["xlsx"])

        if file_upload is not None:
            data = pd.read_excel(file_upload)
            
            st.success('File uploaded successfully!')
            
            RF_pred=predict_model(tuned_rf,data=data)['Label']
            Cat_pred=predict_model(tuned_cat,data=data)['Label']
            Et_pred=predict_model(tuned_et,data=data)['Label']

            NN_input=pd.DataFrame(RF_pred)
            NN_input.columns=['RF']
            NN_input['Cat']=Cat_pred
            NN_input['ET']=Et_pred

            predictions=classifier_ann.predict_classes(NN_input)
            
            data['Prediction']=predictions

            st.write(data)
            st.markdown(get_table_download_link(data), unsafe_allow_html=True)
            



# In[10]:


import base64

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="Load_Predictions.csv">Download csv file</a>'


# In[11]:


if __name__ == '__main__':
    run()


# In[ ]:




