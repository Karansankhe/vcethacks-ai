

--------------------------------EDA------------------------------------
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('EDA-HEALTHCARE.csv')

# Convert the 'ScheduledDay' and 'AppointmentDay' to datetime format
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Convert categorical columns to numeric using label encoding for correlation
df_encoded = df.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'M': 0, 'F': 1})
df_encoded['No-show'] = df_encoded['No-show'].map({'No': 0, 'Yes': 1})

# Exclude non-numeric columns for correlation (dates and strings)
df_encoded = df_encoded.drop(columns=['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood'])

# ---------------- Univariate Non-Graphical Analysis ----------------
# Summary statistics of the dataset
univariate_nongraphical = df.describe(include='all')
print("Univariate Non-Graphical Analysis:")
print(univariate_nongraphical)

# Count the unique values in categorical columns
print("\nGender count:")
print(df['Gender'].value_counts())

print("\nScholarship count:")
print(df['Scholarship'].value_counts())

print("\nNo-show count:")
print(df['No-show'].value_counts())

# ---------------- Multivariate Non-Graphical Analysis ----------------
# Correlation matrix for numerical features after encoding
correlation_matrix = df_encoded.corr()
print("\nMultivariate Non-Graphical Analysis (Correlation Matrix):")
print(correlation_matrix)

# Group by 'No-show' and calculate the mean only for numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns
multivariate_nongraphical = df.groupby('No-show')[numeric_columns].mean()

print("\nGroup by 'No-show' (mean of numeric columns):")
print(multivariate_nongraphical)

# ---------------- Univariate Graphical Analysis ----------------
# Age distribution
plt.figure(figsize=(10,6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Gender distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()

# ---------------- Multivariate Graphical Analysis ----------------
# Plot correlation matrix using a heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot for Age by Gender
plt.figure(figsize=(8,6))
sns.boxplot(x='Gender', y='Age', data=df)
plt.title('Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.xticks([0, 1], ['Male', 'Female'])
plt.show()

# Pairplot for multivariate analysis
sns.pairplot(df_encoded, diag_kind='kde')
plt.suptitle('Pairplot of Numeric Features', y=1.02)
plt.show()

# SMS received vs No-show
plt.figure(figsize=(6,4))
sns.countplot(x='SMS_received', hue='No-show', data=df)
plt.title('SMS Received vs No-show')
plt.show()


---------------------------------diabetes prediction--------------------------
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Extract Features and Targets
X = df_out.drop(columns=['Outcome'])
y = df_out['Outcome']

# Splitting train-test data (80-20 ratio)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Logistic Regression Model
clf_log = LogisticRegression(solver='lbfgs', max_iter=1000)
clf_log.fit(train_X, train_y)
y_pred_log = clf_log.predict(test_X)
log_accuracy = accuracy_score(test_y, y_pred_log)
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")

# Random Forest Model
clf_rf = RandomForestClassifier()
clf_rf.fit(train_X, train_y)
y_pred_rf = clf_rf.predict(test_X)
rf_accuracy = accuracy_score(test_y, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")


--------------------------------diabetes clean-----------------------
# -*- coding: utf-8 -*-
"""Healthcare Data Collection, Cleaning, Integration, and Transformation"""

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(df.head())

# Display descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Check the data types and for any missing values
print("\nData types and missing values:")
print(df.info())

# Check for null values
if df.isnull().values.any():
    print("\nThere are missing values in the dataset.")
else:
    print("\nNo missing values in the dataset.")

# Exploratory Data Analysis
# Visualize the distribution of features
df.hist(bins=10, figsize=(10, 10))
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Countplot of the target variable (Outcome)
sns.countplot(x='Outcome', data=df, palette='Set1')
plt.title('Count of Outcomes')
plt.show()

# Box plot for outlier visualization
sns.set(style="whitegrid")
plt.figure(figsize=(15, 6))
df.boxplot()
plt.title('Box Plot for Feature Distribution')
plt.xticks(rotation=45)
plt.show()

# Remove outliers based on IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Outlier removal
df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Display shapes before and after outlier removal
print(f"Shape before outlier removal: {df.shape}")
print(f"Shape after outlier removal: {df_cleaned.shape}")



---------------------------------cbc NER----------------------------
# Import necessary libraries
import spacy
import pandas as pd

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# Load the dataset (adjust the path as needed)
df = pd.read_csv('cbc.csv')

# Define a function to extract entities from medical text
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Apply the entity extraction function to the medical reports
df['Entities'] = df['long_title'].apply(extract_entities)

# Filter the rows where entities were extracted (i.e., non-empty entities)
df_with_entities = df[df['Entities'].apply(len) > 0]

# Show only the rows where entities are extracted
print(df_with_entities[['subject_id', 'long_title', 'Entities']])

# Save the DataFrame with entities back to a CSV file
output_filename = 'cbc_with_entities.csv'
df_with_entities.to_csv(output_filename, index=False)


--------------------------------------------explain AI-------------------------


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

# Step 1: Create a synthetic healthcare dataset
data = {
    'text': [
        "The doctor was very helpful and attentive during my visit.",
        "I had a terrible experience with the hospital staff.",
        "The treatment was effective and I feel much better now.",
        "The waiting time was too long, and the nurses were rude.",
        "Great service! The team was very professional.",
        "I wouldn't recommend this clinic due to poor service."
    ],
    'sentiment': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Negative']
}

# Create DataFrame and encode sentiments
df = pd.DataFrame(data)
df['label'] = df['sentiment'].map({'Positive': 1, 'Negative': 0})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Step 2: Train a machine learning model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Step 3: Use LIME to explain predictions
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
text_instance = X_test.iloc[0]  # Change index to analyze different samples
exp = explainer.explain_instance(text_instance, model.predict_proba, num_features=5)

# Display explanation and prediction probabilities
exp.show_in_notebook(text=True)
pred_prob = model.predict_proba([text_instance])[0]
print(f"Prediction probabilities for '{text_instance}': Negative: {pred_prob[0]:.4f}, Positive: {pred_prob[1]:.4f}")

# Visualize the explanation
exp.as_pyplot_figure()
plt.title('LIME Explanation for Healthcare Text Classification')
plt.show()





































































import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import google.generativeai as genai
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Configure the Google Gemini model with API key
api_key = os.getenv('GENAI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")

# Function to fetch stock data from Yahoo Finance using yfinance
def fetch_yfinance_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")  # Fetching 1 year of historical data
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol} from Yahoo Finance: {str(e)}")
        return None

# Function to interact with Gemini AI for stock-related queries
def get_gemini_response(symbol):
    prompt_template = (
        ""You are an intelligent assistant with expertise in stock market analysis. Provide a comprehensive analysis of the following stock, including performance trends, financial ratios, and outlook based on recent news. Also, assess the company's dividend history and payout consistency, comparing it with other similar stocks to determine if it offers a better dividend yield."

"Stock: {}"

\n"
        "Analysis:"
    )

    response = model.generate_content(prompt_template.format(symbol), stream=True)
    full_text = ""
    for chunk in response:
        full_text += chunk.text
    return full_text

# Streamlit app setup
st.set_page_config(page_title="Stock Data Dashboard", layout="wide")
st.title("Stock Data Dashboard")

# Streamlit inputs
symbols = st.multiselect("Select Stock Symbols", ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO", 
    "HINDUNILVR.BO", "BHARTIARTL.BO", "ITC.BO", "KOTAKBANK.BO", "SBI.BO",
    "LTI.BO", "WIPRO.BO", "HCLTECH.BO", "M&M.BO", "ADANIGREEN.BO", "NTPC.BO",
    "POWERGRID.BO", "ONGC.BO", "BAJFINANCE.BO", "JSWSTEEL.BO", "HDFC.BO",
    "M&MFIN.BO", "SBILIFE.BO", "CIPLA.BO", "DRREDDY.BO", "SUNPHARMA.BO", "TSLA"], default=["RELIANCE.BO"])

if st.button("Fetch Data"):
    if symbols:
        with st.spinner("Fetching stock data..."):  # Loading spinner
            all_figures = []
            all_data = {}  # To store data for CSV download
            for symbol in symbols:
                data = fetch_yfinance_stock_data(symbol)
                if data is not None and not data.empty:
                    all_data[symbol] = data  # Store data for download
                    st.subheader(f"Stock Data for {symbol}")

                    # Display raw data
                    st.write("**Displaying last 5 records:**")
                    st.write(data.tail())

                    # Create a column layout for visualizations
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Stock Price Chart with Trendline
                        st.subheader(f"Stock Price Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Close'].plot(ax=ax, title=f"Closing Prices for {symbol}", legend=True)

                        # Adding a trendline
                        x = np.arange(len(data))
                        z = np.polyfit(x, data['Close'], 1)  # Linear fit
                        p = np.poly1d(z)
                        ax.plot(data.index, p(x), color='red', linestyle='--', label='Trendline')

                        plt.xlabel("Date")
                        plt.ylabel("Price (USD)")
                        plt.legend()
                        st.pyplot(fig)
                        all_figures.append(fig)

                    with col2:
                        # Volume Traded Chart
                        st.subheader(f"Volume Traded Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Volume'].plot(ax=ax, color='orange', title=f"Volume Traded for {symbol}", legend=True)
                        plt.xlabel("Date")
                        plt.ylabel("Volume")
                        st.pyplot(fig)
                        all_figures.append(fig)

                    with col3:
                        # Moving Average Chart
                        st.subheader(f"Moving Average Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Close'].rolling(window=30).mean().plot(ax=ax, color='blue', label='30-Day Moving Average')
                        data['Close'].plot(ax=ax, title=f"Closing Prices and Moving Average for {symbol}", legend=True)
                        plt.xlabel("Date")
                        plt.ylabel("Price (USD)")
                        plt.legend()
                        st.pyplot(fig)
                        all_figures.append(fig)

                    # Fetch and display Gemini response
                    gemini_response = get_gemini_response(symbol)
                    st.subheader(f"Gemini Response for {symbol}")
                    st.write(gemini_response)

                    # Future Trend Analysis
                    st.subheader(f"Future Trend Analysis for {symbol}")
                    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
                    future_prices = data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.05, size=30)).cumprod()  # Simulated prices
                    future_data = pd.Series(future_prices, index=future_dates)

                    # Plot future trend
                    fig, ax = plt.subplots(figsize=(3, 2))  # Smaller figure size
                    data['Close'].plot(ax=ax, label='Historical Prices', legend=True, fontsize=8)
                    future_data.plot(ax=ax, label='Projected Future Prices', color='green', linestyle='--', legend=True, fontsize=8)

                    plt.title(f"Projected Future Prices for {symbol}", fontsize=10)
                    plt.xlabel("Date", fontsize=8)
                    plt.ylabel("Price (USD)", fontsize=8)
                    plt.legend(fontsize=8)
                    st.pyplot(fig)
                    all_figures.append(fig)

                    # CSV Download
                    csv = data.to_csv().encode()
                    st.download_button(f"Download {symbol} Data as CSV", csv, f"{symbol}_data.csv", "text/csv")

                else:
                    st.error(f"Failed to fetch data for {symbol}")

    else:
        st.error("Please select at least one stock symbol.")
