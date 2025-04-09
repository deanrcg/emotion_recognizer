import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("Note: NLTK resources couldn't be downloaded. If you already have them, this is fine.")

print("🤖 Building Emotion Recognition Model...")

# Create an expanded dataset
emotions_data = {
    'text': [
        # Happy emotions
        "I am so happy today!", "This is the best day ever!", "I'm feeling great about the results.",
        "I love this so much!", "This is absolutely wonderful", "I'm thrilled with the outcome!",
        "What a fantastic experience!", "I'm overjoyed with the news!", "This makes me so happy!",
        "I'm delighted with the progress!", "I'm ecstatic about the results!", "This is amazing!",
        
        # Angry emotions
        "I'm so disappointed with the outcome.", "This makes me so angry and frustrated.",
        "I hate when this happens to me.", "I'm furious about what they did",
        "This is completely unacceptable", "I'm outraged by this behavior!",
        "This is infuriating!", "I'm so mad right now!", "This is ridiculous!",
        "I can't believe this is happening!", "I'm boiling with anger!", "This is outrageous!",
        
        # Sad emotions
        "I'm really sad about what happened.", "That news was heartbreaking.",
        "I feel awful and depressed.", "I miss the old days so much",
        "I'm feeling blue today", "This is so disheartening",
        "I'm feeling down today", "This is really upsetting", "I'm devastated by this news",
        "I'm heartbroken", "This is so depressing", "I'm feeling really low",
        
        # Fear emotions
        "I'm worried about the future.", "I'm anxious about the presentation.",
        "I'm scared this won't work out.", "I'm terrified of what might happen",
        "This is really frightening", "I'm afraid of what's coming",
        "I'm nervous about the outcome", "This is quite alarming", "I'm petrified",
        "I'm feeling anxious", "This is scary", "I'm really concerned",
        
        # Surprise emotions
        "I'm surprised by how well it went!", "Wow! I didn't expect that at all!",
        "That was a shocking turn of events.", "That startled me!",
        "This is unexpected!", "I'm amazed by this!", "This is astonishing!",
        "I'm stunned by the results!", "This is incredible!", "I'm shocked!",
        "This is unbelievable!", "I'm taken aback by this!"
    ],
    'emotion': [
        'happy', 'happy', 'happy', 'happy', 'happy', 'happy',
        'happy', 'happy', 'happy', 'happy', 'happy', 'happy',
        'angry', 'angry', 'angry', 'angry', 'angry', 'angry',
        'angry', 'angry', 'angry', 'angry', 'angry', 'angry',
        'sad', 'sad', 'sad', 'sad', 'sad', 'sad',
        'sad', 'sad', 'sad', 'sad', 'sad', 'sad',
        'fear', 'fear', 'fear', 'fear', 'fear', 'fear',
        'fear', 'fear', 'fear', 'fear', 'fear', 'fear',
        'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
        'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(emotions_data)

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(processed_tokens)
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return ""

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Remove empty processed texts
df = df[df['processed_text'].str.len() > 0]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['emotion'], test_size=0.2, random_state=42, stratify=df['emotion']
)

# Convert text to features using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Perform cross-validation
cv_scores = cross_val_score(classifier, X_train_tfidf, y_train, cv=5)
print(f"\n✅ Model trained successfully!")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f}")

# Evaluate on test set
y_pred = classifier.predict(X_test_tfidf)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))

# Function to predict emotion for new text
def predict_emotion(text):
    if not isinstance(text, str) or not text.strip():
        print("Error: Please provide a valid text input.")
        return None
    
    processed = preprocess_text(text)
    if not processed:
        print("Error: No meaningful text after preprocessing.")
        return None
    
    try:
        tfidf = vectorizer.transform([processed])
        prediction = classifier.predict(tfidf)[0]
        probabilities = classifier.predict_proba(tfidf)[0]
        emotion_prob = dict(zip(classifier.classes_, probabilities))
        sorted_emotions = sorted(emotion_prob.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nAnalyzing: \"{text}\"")
        print(f"📊 Prediction: {prediction.upper()}")
        print("\nConfidence scores:")
        
        # Print a simple visualization of confidence
        for emotion, prob in sorted_emotions:
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{emotion.ljust(8)}: {bar} {prob:.2f}")
        
        return prediction
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

# Interactive loop
print("\n" + "="*60)
print("🔍 INTERACTIVE EMOTION RECOGNIZER")
print("="*60)
print("Type a message to analyze its emotion (or 'quit' to exit)")

while True:
    user_input = input("\n> ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    if not user_input.strip():
        print("Please enter some text to analyze.")
        continue
    
    predict_emotion(user_input)