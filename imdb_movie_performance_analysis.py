
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from collections import Counter

# Load dataset
file_path = '/path/to/IMDB_Movies.csv'  # Replace with actual path to the CSV file
imdb_data = pd.read_csv(file_path)

# Preprocessing
numeric_cols = imdb_data.select_dtypes(include=['float64', 'int64']).columns
imdb_data[numeric_cols] = imdb_data[numeric_cols].fillna(imdb_data[numeric_cols].median())

categorical_cols = imdb_data.select_dtypes(include=['object']).columns
imdb_data[categorical_cols] = imdb_data[categorical_cols].fillna('Unknown')
imdb_data['title_year'] = imdb_data['title_year'].astype('Int64', errors='ignore')

# --- EDA Section ---

# Popularity Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x=imdb_data['movie_facebook_likes'], y=imdb_data['imdb_score'])
plt.title('Movie Facebook Likes vs. IMDB Score')
plt.xlabel('Movie Facebook Likes')
plt.ylabel('IMDB Score')
plt.grid()
plt.show()

# Financial Insights
plt.figure(figsize=(10, 6))
sns.scatterplot(x=imdb_data['budget'], y=imdb_data['gross'])
plt.title('Budget vs. Gross Revenue')
plt.xlabel('Budget')
plt.ylabel('Gross Revenue')
plt.grid()
plt.show()

imdb_data['ROI'] = (imdb_data['gross'] - imdb_data['budget']) / imdb_data['budget']

# Genre Trends
genre_counts = Counter('|'.join(imdb_data['genres']).split('|'))
top_genres = genre_counts.most_common(10)
genre_df = pd.DataFrame(top_genres, columns=['Genre', 'Count'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Genre', data=genre_df)
plt.title('Top 10 Movie Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.grid()
plt.show()

# Director Analysis
top_directors = imdb_data.groupby('director_name')['imdb_score'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top_directors.plot(kind='barh')
plt.title('Top 10 Directors by Average IMDB Score')
plt.xlabel('Average IMDB Score')
plt.ylabel('Director')
plt.grid()
plt.show()

# Yearly Trends
yearly_data = imdb_data.groupby('title_year').agg({
    'budget': 'mean',
    'gross': 'mean',
    'imdb_score': 'mean'
}).reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_data, x='title_year', y='budget', label='Average Budget')
sns.lineplot(data=yearly_data, x='title_year', y='gross', label='Average Gross Revenue')
sns.lineplot(data=yearly_data, x='title_year', y='imdb_score', label='Average IMDB Score')
plt.title('Trends in Budget, Gross, and IMDB Score Over the Years')
plt.xlabel('Year')
plt.ylabel('Value')
plt.grid()
plt.legend()
plt.show()

# Language Analysis
top_languages = imdb_data['language'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_languages.plot(kind='bar', color='blue', alpha=0.7)
plt.title('Top 10 Languages in Movies')
plt.xlabel('Language')
plt.ylabel('Number of Movies')
plt.grid()
plt.show()

# Relationship Between Reviews and IMDB Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x=imdb_data['num_critic_for_reviews'], y=imdb_data['imdb_score'])
plt.title('Critic Reviews vs IMDB Score')
plt.xlabel('Number of Critic Reviews')
plt.ylabel('IMDB Score')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=imdb_data['num_user_for_reviews'], y=imdb_data['imdb_score'])
plt.title('User Reviews vs IMDB Score')
plt.xlabel('Number of User Reviews')
plt.ylabel('IMDB Score')
plt.grid()
plt.show()

# ROI by Genre
imdb_data['ROI'] = (imdb_data['gross'] - imdb_data['budget']) / imdb_data['budget']
genre_roi = imdb_data['genres'].str.split('|').explode().groupby(imdb_data['genres']).mean()
genre_roi_sorted = genre_roi.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_roi_sorted.values, y=genre_roi_sorted.index, orient='h')
plt.title('Top Genres by ROI')
plt.xlabel('ROI')
plt.ylabel('Genre')
plt.grid()
plt.show()

# Revenue Trends by Year
yearly_revenue = imdb_data.groupby('title_year')['gross'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=yearly_revenue['title_year'], y=yearly_revenue['gross'])
plt.title('Revenue Trends Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Gross Revenue')
plt.grid()
plt.show()

# Language-Specific Success
plt.figure(figsize=(10, 6))
sns.boxplot(x='language', y='imdb_score', data=imdb_data)
plt.title('IMDB Scores by Language')
plt.xlabel('Language')
plt.ylabel('IMDB Score')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Content Rating and ROI
content_roi = imdb_data.groupby('content_rating')['ROI'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
content_roi.plot(kind='bar', color='purple', alpha=0.7)
plt.title('ROI by Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('ROI')
plt.grid()
plt.show()

# Top Directors by Revenue
top_directors_revenue = imdb_data.groupby('director_name')['gross'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top_directors_revenue.plot(kind='barh')
plt.title('Top Directors by Average Gross Revenue')
plt.xlabel('Average Gross Revenue')
plt.ylabel('Director')
plt.grid()
plt.show()

# --- Prediction Model ---

# Select relevant features and target
features = [
    'budget', 'duration', 'director_facebook_likes',
    'actor_1_facebook_likes', 'actor_2_facebook_likes',
    'actor_3_facebook_likes', 'num_critic_for_reviews', 'num_user_for_reviews'
]
target = 'imdb_score'

imdb_data_clean = imdb_data.dropna(subset=features + [target])
X = imdb_data_clean[features]
y = imdb_data_clean[target]

for col in features:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')
X.fillna(X.median(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numerical_transformer = StandardScaler()
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, features)])
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=100))])

# Train model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


