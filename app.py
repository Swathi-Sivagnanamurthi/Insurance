from pycaret.regression import load_model,predict_model
import streamlit as st
import pandas as pd
model=load_model('insurance_dt')
input_dict={'age':20,'sex':'male','bmi':20,'children':2,'smoker':'yes','region':'southwest'}
input_df=pd.DataFrame([input_dict])
prediction_df=predict_model(estimator=model,data=input_df)
print(prediction_df)
predictions=prediction_df.iloc[0]['prediction_label']
print(predictions)
st.markdown(predictions)