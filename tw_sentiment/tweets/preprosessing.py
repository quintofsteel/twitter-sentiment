import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

def preprocess_tweets(tweets):
    # Convert tweets data to a Pandas dataframe
    df = pd.DataFrame(tweets)

    # Clean the tweets data
    df['cleaned_tweet'] = df['text'].apply(clean_tweet)

    # Perform sentiment analysis on the tweets data
    df['sentiment_score'] = df['cleaned_tweet'].apply(analyze_sentiment)

    # Return the preprocessed tweets data as a dictionary
    return df.to_dict('records')

def clean_tweet(tweet):
    # Remove URLs, mentions, hashtags, and special characters from tweets
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Tokenize the tweet into words
    tokens = word_tokenize(tweet)

    # Remove stop words and perform stemming or lemmatization
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [stemmer.stem(word) if stemmer.stem(word) not in stop_words else lemmatizer.lemmatize(word) for word in tokens]

    # Join the cleaned tokens back into a tweet
    cleaned_tweet = ' '.join(cleaned_tokens)

    return cleaned_tweet

def analyze_sentiment(tweet):
    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Calculate the sentiment score of the tweet
    sentiment_score = sia.polarity_scores(tweet)['compound']

    return sentiment_score
