import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import pandas as pd

#Load the trained model

model = tf.keras.models.load_model('model.keras')

#load encoders and scaler

with open('OneHot_Encoder.pkl','rb') as file:
    OneHot_Encoder = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#User input
geography = st.selectbox('Geography',OneHot_Encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,99)
balance = st.number_input('Balance')
Credit_Score = st.number_input('Credit Score')
Estimated_Salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
Num_of_Products = st.slider('Number of products',1,4)
Has_Cr_Card = st.selectbox('Has Credit Card',[0,1])
Is_active_member = st.selectbox('Is active member',[0,1])

user_input_df = pd.DataFrame({
    'CreditScore' : [Credit_Score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [Num_of_Products],
    'HasCrCard' : [Has_Cr_Card],
    'IsActiveMember' : [Is_active_member],
    'EstimatedSalary' : [Estimated_Salary]
    })

#One Hot Encode 'Geography'

OneHot_Encoder_geo = OneHot_Encoder.transform([[geography]]).toarray()
OneHot_Encoder_geo_df = pd.DataFrame(OneHot_Encoder_geo,columns=OneHot_Encoder.get_feature_names_out(['Geography']))
 
#Concat One Hot encoder data

user_input= pd.concat([user_input_df.reset_index(drop=True),OneHot_Encoder_geo_df],axis=1)

#Scale the input data

scaled_input = scaler.transform(user_input)

#churn prediction

prediction = model.predict(scaled_input)

#prediction propability

prediction_prob = prediction[0][0]

st.write("Probability:",prediction_prob)

if prediction_prob > 0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is unlikely to churn")