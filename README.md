# Loan Approval Prediction

This project predicts whether a loan application should be approved or rejected using machine learning.


## Project Overview

The model is trained on customer financial and demographic data. 
After training and evaluation, the best-performing model is deployed as a REST API using FastAPI and
further dockerized for production-ready usage.

## Key Features
- Data preprocessing and feature engineering
- Machine learning model training and evaluation
- REST API for real-time predictions using FastAPI
- Docker-based containerized deployment

## Input Features
- Number of dependents  
- Education  
- Self-employed status  
- Annual income  
- Loan amount  
- Loan term  
- CIBIL score  
- Movable assets  
- Immovable assets  

## Output
- Loan approval decision (Approved / Rejected)  
- Probability of loan approval  

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- FastAPI  
- Docker  

## Author
Shivam Pandey
