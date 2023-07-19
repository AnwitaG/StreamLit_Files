#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd


# In[2]:


model = pickle.load(open('model.pkl','rb'))


# In[3]:


def predict_price(carat, cut, color, clarity, depth, table,x, y, z):
    input=np.array([[carat, cut, color, clarity, depth, table,x, y, z]]).astype(np.float64)
    prediction = model.predict(input)
    
    return int(prediction)


# In[4]:


def main():
    st.title("Diamond Price Prediction (o8)")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Diamond Price Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)


# In[5]:


carat = st.text_input("carat","Type Here")
cut = st.text_input("cut","Type Here")
color = st.text_input("colur","Type Here")
clarity = st.text_input("clarity","Type Here")
depth = st.text_input("depth","Type Here")
table = st.text_input("table","Type Here")
x = st.text_input("x","Type Here")
y = st.text_input("y","Type Here")
z = st.text_input("z","Type Here")


# In[6]:


safe_html ="""  
        <div style="background-color:#80ff80; padding:10px >
        <h2 style="color:white;text-align:center;"> Price of a Diamond</h2>
        </div>
        """
if st.button("Predict the price"):
    output = predict_price(carat, cut, color, clarity, depth, table,x, y, z)
    st.success('The price is {}'.format(output))


# In[7]:


if __name__=='__main__':
    main()

