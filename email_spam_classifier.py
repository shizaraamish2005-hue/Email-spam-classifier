import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Small built-in stopword list (no internet needed)
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'to', 'of', 'in', 'on', 'for', 'with', 'at', 'by', 'from', 'this',
    'that', 'it', 'as', 'be', 'i', 'you', 'your', 'we', 'they', 'he', 'she'
}

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words)

data['message'] = data['message'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

msg = input("\nEnter a message: ")
msg = clean_text(msg)
msg = vectorizer.transform([msg])
prediction = model.predict(msg)

print("Spam Message" if prediction[0] == 1 else "Not Spam")
