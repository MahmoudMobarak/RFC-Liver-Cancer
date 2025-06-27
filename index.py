import streamlit as st
import streamlit_shadcn_ui as ui
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
with open(r'C:\Users\mahmo\Desktop\VS\Liver_Cancer_rfc\liver_cancer_rfc.pkl', 'rb') as f:
    RFC = pickle.load(f)
feature_name=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']
st.markdown('# Hello!')
st.markdown('### This model will help pridict the probabilty that you have liver cancer using some information about you')
radio_options = [
    {"label": "Female", "value": "1", "id": "f1"},
    {"label": "Male", "value": "0", "id": "m0"},
]
radio_value = ui.radio_group(options=radio_options, default_value="1", key="radio1")
age=st.number_input("Please enter your age", 0, 100)
Tota_Bilirubin=st.number_input("Please enter your Total Bilirubin", 0, 75,key=1)
direct_Bilirubin=st.number_input("Please enter your direct Bilirubin", 0.1, 19.7,key=2)
Alkaline_Phosphotase=st.number_input("Please enter your Alkaline Phosphotase", 63, 2110,key=3)
Alamine_Aminotransferase=st.number_input("Please enter your Alamine Aminotransferase", 10, 2000,key=4)
Aspartate_Aminotransferase=st.number_input("Please enter your Aspartate Aminotransferase", 10, 4292,key=5)
Total_Protiens=st.number_input("Please enter your Total Protiens", 2.7, 9.6,key=7)
Albumin=st.number_input("Please enter your Albumin", 0.9,5.5,key=8)
Albumin_and_Globulin_Ratio=st.number_input("Please enter your Albumin and Globulin Ratio", 0.3,2.8,key=9)
data_values=[age,radio_value,Tota_Bilirubin,direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]
data_df = pd.DataFrame([data_values], columns=feature_name)
Columns=data_df.columns.tolist()
Columns.remove('Gender')
for i in range(len(Columns)):
  if data_df[Columns[i]].mean()>1:
    data_df[Columns[i]]=StandardScaler().fit_transform(data_df[[Columns[i]]])
predict=st.button("Predict!")
if predict==True:
    prob=RFC.predict_proba(data_df)
    pred_prob=prob[:, 1] * 100
    st.markdown(f'### There is a {pred_prob}% chance you have liver cancer')
    st.write('Therefor....')
    predict=RFC.predict(data_df)
    if predict==1:
        st.markdown('### There is a high chance you have liver cancer')
    elif predict ==0:
        st.markdown('### There is a high chance you dont have liver cancer')




