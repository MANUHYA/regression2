import streamlit as st
import os
import pandas as pd
import joblib as jb


def return_df(age,sex,bmi,children,smoker,region):
	kbn={
	'age':[age],
	'sex':[sex],
	'bmi':[bmi],
	'children':[children],
	'smoker':[smoker],
	'region':[region]
	}
	final_df=pd.DataFrame(kbn)
	return final_df

def base_model():
	bmodel=jb.load(os.path.join('finalised_model_regression.pkl'))
	return bmodel

st.markdown(f'''
<h1 align='center' >Medical Cost</h1>
''',unsafe_allow_html=True)
age=st.number_input('Enter your age',min_value=0)
sex=st.selectbox('Select your gender',['male','female'])
bmi=st.number_input('bmi',min_value=0)
children=st.number_input('Count of your Children',min_value=0)
smoker=st.selectbox(' Do you smoke',['yes','no'])
region=st.selectbox('Select your Region',['southeast','southwest','northeast','northwest'])

df=return_df(age,sex,bmi,children,smoker,region)
if st.button('Get your charges'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	st.write('Individual medical cost billed by health insurance is: {}$'.format(predictions))


