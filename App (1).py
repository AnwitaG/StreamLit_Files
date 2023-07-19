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


df = pd.read_csv('DiamondsPrices.csv')


# In[4]:


df.loc[df["cut"] == 'Fair', "cut"] = 0
df.loc[df["cut"] == "Good", "cut"] = 1
df.loc[df["cut"] == "Very Good", "cut"] = 2
df.loc[df["cut"] == "Ideal", "cut"] = 3
df.loc[df["cut"] == "Premium", "cut"] = 4
df['cut'] = df['cut'].astype(float)
df


# In[5]:


df.loc[df["color"] == 'E', "color"] = 0
df.loc[df["color"] == "I", "color"] = 1
df.loc[df["color"] == "J", "color"] = 2
df.loc[df["color"] == "H", "color"] = 3
df.loc[df["color"] == "F", "color"] = 4
df.loc[df["color"] == "G", "color"] = 5
df.loc[df["color"] == "D", "color"] = 6
df['color'] = df['color'].astype(float)
df


# In[6]:


df.loc[df["clarity"] == 'SI2', "clarity"] = 0
df.loc[df["clarity"] == "SI1", "clarity"] = 1
df.loc[df["clarity"] == "VS1", "clarity"] = 2
df.loc[df["clarity"] == "VS2", "clarity"] = 3
df.loc[df["clarity"] == "VVS2", "clarity"] = 4
df.loc[df["clarity"] == "VVS1", "clarity"] = 5
df.loc[df["clarity"] == "I1", "clarity"] = 6
df.loc[df["clarity"] == 'IF', "clarity"] = 7
df['clarity'] = df['clarity'].astype(float)

df


# In[7]:


df.dtypes


# In[8]:


def predict_price(carat, cut, color, clarity, depth, table,x, y, z):
    input=np.array([[carat, cut, color, clarity, depth, table,x, y, z]]).astype(np.float64)
    prediction = model.predict(input)
    
    return int(prediction)


# In[9]:


def main():
    st.title("Diamond Price Prediction (o8)")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Diamond Price Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)


# In[10]:


df.columns


# In[11]:


carat = st.text_input("carat","Type Here")
cut = st.text_input("cut","Type Here")
color = st.text_input("colur","Type Here")
clarity = st.text_input("clarity","Type Here")
depth = st.text_input("depth","Type Here")
table = st.text_input("table","Type Here")
x = st.text_input("x","Type Here")
y = st.text_input("y","Type Here")
z = st.text_input("z","Type Here")


# In[12]:


safe_html ="""  
        <div style="background-color:#80ff80; padding:10px >
        <h2 style="color:white;text-align:center;"> Price of a Diamond</h2>
        </div>
        """
if st.button("Predict the price"):
    output = predict_price(carat, cut, color, clarity, depth, table,x, y, z)
    st.success('The price is {}'.format(output))


# In[13]:


if __name__=='__main__':
    main()

