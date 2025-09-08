 Spam_Email_Detection
📌 Overview

This project detects whether an email is Spam or Ham (Not Spam) using machine learning. It applies text preprocessing, feature extraction, and classification techniques to build a spam filter.

⚙️ Features

Loads and preprocesses email dataset (spam.csv).

Uses Bag of Words (CountVectorizer) for feature extraction.

Trains a Naive Bayes classifier (MultinomialNB).

Evaluates the model with accuracy, precision, recall, F1-score, and ROC curve.

Visualizes email text with WordClouds.

🛠️ Technologies Used

Python 3.x

NumPy, Pandas → Data handling

Matplotlib, Seaborn → Data visualization

WordCloud → Text visualization

Scikit-learn → Machine Learning (CountVectorizer, Naive Bayes, Pipeline, Metrics)

📂 Project Structure
Email_Spam_Detection_with_Machine_Learning.ipynb   # Main Jupyter Notebook
spam.csv                                           # Dataset (loaded from GitHub in code)
README.md                                          # Project documentation

🔧 Installation

Clone the repository or download the notebook.

git clone https://github.com/yourusername/spam-email-detector.git
cd spam-email-detector


Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn wordcloud


Run the notebook:

jupyter notebook Email_Spam_Detection_with_Machine_Learning.ipynb

▶️ Usage

Open the Jupyter Notebook.

Run the cells step by step.

The notebook will:

Load and preprocess the dataset.

Train a Naive Bayes spam classifier.

Display evaluation metrics and visualizations.

Allow predictions for new email text.

📊 Results

The Naive Bayes model achieved high accuracy (~97%).

WordClouds showed common words in spam vs ham emails.

Confusion matrix and ROC curve visualized model performance.

📌 Future Work

Use TF-IDF or word embeddings instead of CountVectorizer.

Try other classifiers (SVM, Random Forest, Logistic Regression).

Deploy as a web app using Flask or Streamlit.

👩‍💻 Author

Developed as part of Oasis Infobyte Internship – Task 4
