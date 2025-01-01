# IMDB-Movie-Analysis-Project-

Project Overview
This project involves analyzing a dataset of over 5,000 IMDB movie records to predict movie performance based on various features such as budget, duration, and social media popularity. The objective is to uncover trends that influence IMDB scores and build a predictive model to forecast movie performance.

Key Features
Dataset: IMDB movies dataset with over 5,000 records.
Primary Target: IMDB movie scores (rating).
Features: Budget, duration, number of reviews, social media likes for directors and actors, etc.
Technologies Used
Python
Pandas
Matplotlib
Seaborn
Scikit-learn
Project Steps
1. Data Loading & Preprocessing
Loaded the dataset using Pandas.
Cleaned the dataset by handling missing values:
For numerical columns: filled missing values with the median.
For categorical columns: replaced missing values with 'Unknown'.
Converted the title_year column to Int64 and ensured data consistency for analysis.
2. Exploratory Data Analysis (EDA)
Performed in-depth EDA to identify key patterns and correlations.
Popularity Analysis: Analyzed the relationship between movie Facebook likes and IMDB scores.
Financial Insights: Investigated the correlation between budget and gross revenue.
Genre Trends: Visualized the most popular movie genres.
Director Analysis: Examined the relationship between directors and average IMDB scores.
Yearly Trends: Analyzed movie performance trends (budget, gross, IMDB scores) over the years.
Language Analysis: Visualized the impact of movie language on IMDB scores.
3. Visualization
Used Matplotlib and Seaborn to create visualizations for:
Budget vs. Gross Revenue
IMDB Score vs. Critic and User Reviews
Trends in movie genres and languages
Director performance and year-over-year analysis
4. Prediction Model
Selected relevant features (budget, duration, director/actor likes, reviews) for predicting IMDB scores.
Random Forest Regressor was implemented to build a predictive model:
Trained the model with the cleaned dataset.
Evaluated the model using R² Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
Achieved an R² score of 0.85, indicating a strong prediction accuracy.
5. Evaluation
Evaluated model performance using:
R² Score: 0.85 (indicating that the model explains 85% of the variance in IMDB scores).
MAE and MSE metrics provided additional insights into prediction errors.
Results
Best Performing Model: Random Forest Regressor with an R² score of 0.85.
Key Insights:
Movies with higher budgets tend to generate higher gross revenues.
Genre and director performance are strong predictors of IMDB scores.
Social media presence (director/actor likes) has a modest impact on IMDB scores.
Conclusion
This project demonstrates the use of machine learning for predicting movie performance and reveals valuable insights into the factors that influence IMDB scores. The Random Forest model provides a reliable prediction mechanism with an R² score of 0.85, and the EDA provides actionable insights for movie studios and stakeholders.
