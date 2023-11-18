## Overview
Machine failures can lead to significant downtime and maintenance costs.

This project aims to address this issue by building a machine learning model that predicts failures in advance leading to extended equipment lifespan and overcoming the challenges related to machine downtime including Long Maintenance Waiting time & Productivity Decrease

Anticipating machine failures leads to making informed decisions on production planning and resource allocation.
This approach empowers businesses to proactively manage maintenance, minimize downtime, and optimize operational efficiency

## Built with:

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-blue?style=flat&logo=scikit-learn)](https://scikit-learn.org/) 
[![Azure SDK v2](https://img.shields.io/badge/Azure%20SDK%20v2-Latest-blue?style=flat&logo=microsoft-azure)](https://azure.microsoft.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Latest-blue?style=flat&logo=mlflow)](https://mlflow.org/)


## Key Features
- Proactive Maintenance: Identify potential machine failures before they occur, allowing for proactive maintenance.
- Operational Efficiency: Optimize production schedules and resource allocation by minimizing unexpected downtime.
- Scalable Model Management: Utilize MLflow for efficient model tracking, versioning, and deployment.
- User-friendly Interface: Easy-to-use scripts for training models, tracking experiments & evaluating results.

## Approach
This project focuses on predicting machine failures using Random Forests, implemented with Scikit-learn.
The Machine Learning Lifecycle was managed with MLflow. 
The goal is to create a robust model that can accurately predict machine failures based on input features representing the states of the componenets of a machine.

### Dataset Overview
<h3 align="center">Dataset Sample relavant for feature overview</h3>
<p align="center">
  <img src="imgs/data.jpg" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>

<h3 align="center">Summary Statistics allow understanding of features</h3>
<p align="center">
  <img src="imgs/statistics.jpg" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>

<h3 align="center">Box Plots allow the choice of an Appropriate Normalization Method</h3>
<p align="center">
  <img src="imgs/BoxPlots.png" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>

<h3 align="center">Kernel Density - Histogram Estimation</h3>
<div align="center">
  <img src="imgs/kde_1.png" alt="Dataset Train & Validation" style="width:25%; height:auto; margin: 10px;">
  <img src="imgs/kde_2.png" alt="Dataset Train & Validation" style="width:25%; height:auto; margin: 10px;">
  <img src="imgs/kde_3.png" alt="Dataset Train & Validation" style="width:25%; height:auto; margin: 10px;">
  <img src="imgs/kde_4.png" alt="Dataset Train & Validation" style="width:25%; height:auto; margin: 10px;">
</div>


<h3 align="center">Pearson Correlation Analysis to Understand Linear Relationship between features</h3>
<p align="center">
  <img src="imgs/correlation_matrix.png" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>

<h3 align="center">Spearman Correlation Analysis to Understand Non Linear Relationship between features</h3>
<p align="center">
  <img src="imgs/Spearman_CorrelationMatrix.png" alt="Dataset Train & Validation" style="width:50%; height:auto;">
</p>

<h2 align="center">Evaluation: Accuracy of: 0.9937  </h2>

<table align="center">
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
    <th>Support</th>
  </tr>
  <tr>
    <td>Class 0</td>
    <td>0.9935</td>
    <td>1.0</td>
    <td>0.9967</td>
    <td>2907.0</td>
  </tr>
  <tr>
    <td>Class 1</td>
    <td>1.0</td>
    <td>0.7957</td>
    <td>0.8862</td>
    <td>93.0</td>
  </tr>
  <tr>
    <td>Macro Avg</td>
    <td>0.9968</td>
    <td>0.8978</td>
    <td>0.9415</td>
    <td>3000.0</td>
  </tr>
  <tr>
    <td>Weighted Avg</td>
    <td>0.9937</td>
    <td>0.9937</td>
    <td>0.9933</td>
    <td>3000.0</td>
  </tr>
</table>

## Setup
- Clone the repository & navigate
- Install dependencies
- View Exploratory Data analysis on EDA.ipynb
- Model Training
~~~
python main.py
~~~

## Contact:
Feel free to reach out to me on LinkedIn or through email & don't forget to visit my portfolio.
 
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/samiabelhaddad/)
  [![Email](https://img.shields.io/badge/Email-Contact%20Me-brightgreen?style=flgat&logo=gmail)](mailto:samiamagbelhaddad@gmail.com)
  [![Portfolio](https://img.shields.io/badge/Portfolio-Visit%20My%20Portfolio-white?style=flat&logo=website)](https://samiabelhaddad.me/)








