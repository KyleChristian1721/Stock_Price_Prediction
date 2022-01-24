from altair.vegalite.v4.schema.channels import Href
from scipy.sparse import data
import streamlit as st
import streamlit.components as stc
import base64 
import numpy as np
import pandas as pd
import time
import plotly.figure_factory as ff
import altair as alt
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
from datetime import date
from sklearn.svm import SVR
from plotly import graph_objs as go
from PIL import Image

# st.set_page_config(layout="wide")
timestr = time.strftime("%Y%m%d-%h%M%S")

st.markdown("<h2 style='text-align: center;'>STOCK PRICE PREDICTION USING SVR <br> SM Mall Price in Manila Philippines</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h6 style='color: green; margin-left: 30px'>Updated as of 17/01/2022: </h6>", unsafe_allow_html=True)

# Raw data Chart
@st.cache
def load_stock_data(nrows):
    data = pd.read_csv("C:/Users/KyleC/Documents/Datasets/Dislay Datasets.csv", nrows = nrows)
    return data
stock_data = load_stock_data(30)
chart_data = pd.DataFrame(stock_data[:30], columns = [' Close', ' Open', 'low'])
st.area_chart(chart_data)

df = pd.read_csv ("C:/Users/KyleC/Documents/Datasets/Display Datasets.csv")
st.write(df)


# Getting the dataset
df = pd.read_csv ("C:/Users/KyleC/Documents/Datasets/Final Dataset.csv")

# Making The Data set Downloadable
st.download_button (label='Download CSV', data = df.to_csv(), mime='text/csv')

# Renaming of Columns to Avoid Space Error
df = df.rename(columns={' Open': 'Open'})
df = df.rename(columns={' High': 'High'})
df = df.rename(columns={' Low': 'Low'})
df = df.rename(columns={' Close': 'Close'})
df = df.rename(columns={' Adj': 'Adj'})
df = df.rename(columns={' Volume ': 'Volume'})


# Getting The head of the data set
actual_price = df.head(1)
# prepare the data for training for SVR
df.tail(len(df)-1)

# Create Empty lists to store the independent and dependent data
days = list()
adj_close_prices = list()

# Get the dates and Adj Close prices
df_days = df.loc[:, 'Date']
df_adj_close = df.loc[:, 'Adj']

# Create The independent Data set
for day in df_days:
  days.append([int(day.split('/')[1])])

# Create The Dependent data set
for adj_close_price in df_adj_close:
  adj_close_prices.append(float(adj_close_price)) 

# Create The 3 SVR models
lin_svr = SVR (kernel = 'linear', C=1000.0)
lin_svr.fit(days, adj_close_prices)

poly_svr = SVR (kernel = 'poly', C=1000.0, degree = 2)
poly_svr.fit(days, adj_close_prices)

rbf_svr = SVR (kernel = 'rbf', C=1000.0, gamma = 0.15)
rbf_svr.fit(days, adj_close_prices)

# plot the models on a graph
plt.figure(figsize=(16,8))
plt.scatter(days, adj_close_prices, color = 'red', label = 'data')
plt.plot(days, rbf_svr.predict(days), color = 'yellow', label = 'RBF Model')
plt.plot(days, poly_svr.predict(days), color = 'orange', label = 'Polynomial Model')
plt.plot(days, lin_svr.predict(days), color = 'blue', label = 'Linear Model')
plt.legend()
# plt.show()
# st.pyplot()

# To ignore unnecessary warnings from python
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h3 style='text-align: center; margin-bottom:-50px;'>Predicted closed Stock prices for: 17/01/2022</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Show The predicted price for the given day
day = [[31]]
st.markdown("The RBF SVR Predicted: " + str(rbf_svr.predict(day))) 
st.markdown("The Linear SVR Predicted: "+ str(lin_svr.predict(day)))
st.markdown("The polynomial SVR Predicted: " + str(poly_svr.predict(day)))

st.markdown("<h3 style='text-align: center; margin-bottom:-50px;'>Percentage Error for today's forecast:</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h6> Percentage error for RBF = 13.61% (safe)</h6>",unsafe_allow_html=True)
st.markdown("<h6 style='color: red;'> Percentage error for Linear SVR = 8.34% (overfitting)</h6>",unsafe_allow_html=True)
st.markdown("<h6 style='color: red;'> Percentage error for Polynomial SVR = 9% (overfitting)</h6>",unsafe_allow_html=True)
st.caption("<h6 style='text-align: center;  margin-bottom:-50px;'>Lower is better (Note: This depends on the percentage error, if the value is less than 10% its called overfitting which is not safe but more than 10% is safe)</h6>", unsafe_allow_html=True)
st.caption("<hr>", unsafe_allow_html=True)


# Designs and extras
st.markdown("<h2 style='text-align: center;'>What is SVR or Support Vector Regression?</h2>",unsafe_allow_html=True)
st.markdown("<p style ='text-align: justify;'>The Support Vector Regression (SVR) uses the same ideas as the SVM for classification, with a few small differences. For starters, because output is a real number, it becomes incredibly difficult to forecast the information at hand, which has an infinite number of possibilities. </p>",unsafe_allow_html=True)
st.markdown("<p style ='text-align: justify;'>A margin of tolerance (epsilon) is supplied in the case of regression as an approximate estimate to the SVM that the issue would have already requested. Apart from that, there is a more challenging reason: the algorithm is more complex, thus it must be considered. However, the basic idea remains the same: to reduce error by customising the hyperplane to maximise the margin while keeping in mind. </p>",unsafe_allow_html=True)
st.markdown("<p style ='text-align: justify;'>The introduction of a -insensitive zone around the function, known as the -tube, allows SVM to be generalised to SVR. The optimization problem is reformulated in this tube to discover the tube that best approximates the continuous-valued function while balancing model complexity and prediction error. </p>",unsafe_allow_html=True)
st.markdown("<p style ='text-align: justify;'>SVR is defined as an optimization problem by first constructing a convex-insensitive loss function to be reduced and then determining the flattest tube that includes the majority of the training cases. As a result, the loss function and the geometrical parameters of the tube are combined to form a multiobjective function.</p>",unsafe_allow_html=True)
st.markdown("<p style ='text-align: justify;'>Then, using suitable numerical optimization methods, the convex optimization, which has a unique solution, is solved. Support vectors, which are training samples that fall outside the tube's perimeter, are used to represent the hyperplane. </p>",unsafe_allow_html=True)
st.markdown("<p style ='text-align: justify;'>In a supervised-learning environment, the support vectors are the most influential cases that determine the form of the tube, and the training and test data are assumed to be independent and identically distributed (iid), obtained from the same fixed but unknown probability data distribution function.</p>",unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Sample SVR Accuracy chart</h3>",unsafe_allow_html=True)
image = Image.open("C:/Users/KyleC/Documents/Datasets/Pictures/Accuracy chart (25).png")
st.image(image, caption='Accuracy chart with 25 data')

st.subheader("Conclusion")
st.markdown("<p style ='text-align: justify;'>SVR really proves to be better than deep learning methods in cases of limited datasets and also require much less time than its counterpart. In comparison with other regression algorithms, SVR uses much less computation and has high accuracy and credibility.</p>",unsafe_allow_html=True)

# Removing the watermark in the footer
hide_menu_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

