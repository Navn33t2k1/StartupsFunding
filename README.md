📊 Indian Startups Funding Analysis & Recommendation System
📌 Project Overview

This project provides an end-to-end analysis of the Indian startup funding ecosystem, combining data cleaning, exploratory analysis, interactive visualization, and recommendation systems.
It enables insights for both startups and investors through clustering-based recommendations and an interactive dashboard.

🎯 Objectives

Analyze startup funding trends across time, cities, industries, and funding stages

Build an interactive dashboard for dynamic data exploration

Improve data quality using fuzzy string matching

Develop recommendation systems for:

Investors (to find similar investment opportunities)

Startups (to identify comparable startups and competitors)

🗂 Dataset Description

The dataset contains funding information for Indian startups.

Key Columns:

StartUp – Startup name

Investor – Investor(s) involved

Vertical – Industry category

SubVertical – Sub-sector

City – Startup location

Round – Funding stage

Amount in Cr – Funding amount

Date, Year, Month – Time-related features

🧹 Data Cleaning & Preprocessing

Standardized startup and investor names

Applied TheFuzz (fuzzy string matching) to detect and merge similar textual entries

Normalized funding rounds, verticals, and city names

Handled missing, duplicate, and inconsistent values

Converted date fields into proper datetime format

✔ Resulted in a high-quality, analysis-ready dataset

📈 Exploratory Data Analysis (EDA)

Analysis includes:

Year-wise funding trends

City-wise and industry-wise funding distribution

Top investors and funded startups

Funding round patterns

Temporal and seasonal insights

📊 Interactive Dashboard

Built using Streamlit

Visualizations created using Plotly

Features:

Filters by year, city, industry, funding round

Interactive charts and tables

Real-time data exploration

🤖 Recommendation Systems
🔹 Startup Recommendation System

Implemented using KMeans clustering

Groups startups based on:

Funding amount

Industry vertical

Funding frequency

Growth-related features

Helps identify similar startups for benchmarking and competitive analysis

🔹 Investor Recommendation System

Uses clustering to group investors based on:

Investment patterns

Preferred funding stages

Industry focus

Recommends relevant startups to investors and similar investors for analysis

🛠 Tools & Technologies

Python

Pandas & NumPy

Plotly – Interactive visualizations

Streamlit – Dashboard development

TheFuzz – Fuzzy string matching

Scikit-learn (KMeans) – Clustering & recommendations

Jupyter Notebook

📌 Key Insights

Bengaluru dominates the Indian startup funding ecosystem

Fintech and Edtech attract the highest investments

Early-stage funding rounds are most common

Clustering reveals distinct startup and investor profiles
