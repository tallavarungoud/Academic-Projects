import tweepy
from textblob import TextBlob

# --- Twitter API v2 credentials ---
bearer_token = "AAAAAAAAAAAAAAAAAAAAAP434AEAAAAAx8QkEVHMmIVbv9IzNebAsQ9H4gE=jDp00AUJozOK8UzbK98VODWAHUNZXiUlx33jFwVd5B09t0fT5q"

# --- Authenticate using Tweepy Client (API v2) ---
client = tweepy.Client(bearer_token=bearer_token)

# --- Function to fetch tweets ---
def get_tweets(query, max_results=50):
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=['lang']
        )
        if response.data:
            # Keep only English tweets
            return [tweet.text for tweet in response.data if tweet.lang == 'en']
        else:
            print("No tweets found for this query.")
            return []
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

# --- Function to analyze sentiment ---
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# --- Main function ---
def main():
    query = "kfc"  # 🔹 Topic changed to KFC
    tweets = get_tweets(query)

    positive, neutral, negative = 0, 0, 0

    for tweet in tweets:
        sentiment = analyze_sentiment(tweet)
        print(f"Tweet: {tweet}\nSentiment: {sentiment}\n")

        if sentiment == 'positive':
            positive += 1
        elif sentiment == 'neutral':
            neutral += 1
        else:
            negative += 1

    total = positive + neutral + negative
    if total > 0:
        print(f"\nSentiment Summary for '{query}':")
        print(f"Positive: {(positive/total)*100:.2f}%")
        print(f"Neutral: {(neutral/total)*100:.2f}%")
        print(f"Negative: {(negative/total)*100:.2f}%")
    else:
        print("No tweets available for sentiment analysis.")

# --- Correct main execution check ---
if __name__ == "__main__":
    main()
 