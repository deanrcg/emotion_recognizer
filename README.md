# Emotion Recognizer

A Python-based emotion recognition system that uses AI to analyze text and predict emotional content. The system can identify five basic emotions: happiness, anger, sadness, fear, and surprise.

## AI Components

The project uses several AI and machine learning techniques:

### 1. Natural Language Processing (NLP)
- Uses NLTK (Natural Language Toolkit) for text preprocessing
- Tokenizes text into words
- Removes stopwords (common words that don't carry much meaning)
- Uses lemmatization to reduce words to their base form
- This preprocessing helps the AI model better understand the text structure

### 2. Feature Extraction using TF-IDF
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Converts text into numerical features that the AI can process
- Weights words based on their importance in the text
- Creates a feature matrix that represents the text in a way the AI can understand

### 3. Machine Learning Classification
- Uses a Multinomial Naive Bayes classifier
- Trained on a dataset of labeled emotional text examples
- Learns patterns between text features and emotions
- Can predict emotions in new, unseen text

### 4. Cross-Validation and Performance Metrics
- Uses 5-fold cross-validation to evaluate model performance
- Provides confidence scores for predictions
- Shows precision, recall, and F1-score metrics
- Helps ensure the AI model is reliable and accurate

## AI Pipeline

The AI pipeline works as follows:
1. Text input → Preprocessing (NLP)
2. Preprocessed text → Feature extraction (TF-IDF)
3. Features → Classification (Naive Bayes)
4. Classification → Emotion prediction with confidence scores

## Training Data

The AI model was trained on a diverse dataset of 60 examples (12 per emotion) covering:
- Happy emotions
- Angry emotions
- Sad emotions
- Fear emotions
- Surprise emotions

This training allows the model to learn patterns in language that indicate different emotions, making it capable of analyzing new text and predicting the emotional content with reasonable accuracy.

## Features

- Text preprocessing using NLTK
- Emotion classification using TF-IDF and Naive Bayes
- Interactive command-line interface
- Cross-validation and performance metrics
- Support for multiple emotions (happy, angry, sad, fear, surprise)

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk

## Installation

1. Clone the repository:
```bash
git clone https://github.com/deanrcg/emotion_recognizer.git
cd emotion_recognizer
```

2. Install the required packages:
```bash
pip install pandas numpy scikit-learn nltk
```

3. Download required NLTK resources:
```bash
python download_nltk.py
```

## Usage

Run the emotion recognizer:
```bash
python emotion_recognizer.py
```

Type any text when prompted, and the system will analyze its emotional content. The system will display:
- The predicted emotion
- Confidence scores for each possible emotion
- A visual representation of the confidence levels

To exit the program, type 'quit'.

## Example

```
> I'm really excited about this project!
📊 Prediction: HAPPY

Confidence scores:
happy    : ████████████████░░░░ 0.82
surprise : ████░░░░░░░░░░░░░░░░ 0.12
fear     : ░░░░░░░░░░░░░░░░░░░░ 0.03
sad      : ░░░░░░░░░░░░░░░░░░░░ 0.02
angry    : ░░░░░░░░░░░░░░░░░░░░ 0.01
```

## License

MIT License 