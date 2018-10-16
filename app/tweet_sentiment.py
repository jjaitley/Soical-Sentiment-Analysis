import tweepy
import re
import pickle

from tweepy.auth import OAuthHandler

#from tweepy import oAuthHandler

#Initializing the keys
consumer_key ='9d0KWvb4v4wc3XPliFbuT9Tje'
consumer_secret ='BdwGrkL0jyWNecNqnTEJUfQzrq0Ahya91cYX01QH0sXH1LzEsb'
access_token ='177558897-7q8gdaRktZSK2YQyL74ai24w8OHlOUN56wwfCpJb'
access_secret ='al7K29BmxS70b6pI59PkriFvnFuPbQBd5IV6LIHB4wXSA'

auth =   OAuthHandler(consumer_key,consumer_secret) # verifying the authenticity of app Every app u build on twitter has unique cons key & secret
auth.set_access_token(access_token,access_secret) # This gives the right to fetch tweets from twitter

args=['laden']
api= tweepy.API(auth,timeout=10)

list_tweets= []

query=args[0]

if len(args) == 1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent').items(100):
        list_tweets.append(status.text)

#Loading our model the vectorizer and the classifier        
with open('tfidfmodel.pickle','rb') as f:
    vectorizer=pickle.load(f)

with open('classifier.pickle','rb')  as f:
    clf = pickle.load(f)

#testing my classifier     
#clf.predict(vectorizer.transform(['have a good life']))    
total_pos = 0
total_neg = 0

for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    sent  = clf.predict(vectorizer.transform([tweet]).toarray())
    #print(tweet, "::", sent)   
    if sent[0] == 1:
        total_pos += 1
    else:
        total_neg += 1
    
# Visualizing the results
import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive','Negative']
y_pos = np.arange(len(objects))

plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
plt.xticks(y_pos,objects)
plt.ylabel('Number')
plt.title('Number of Postive and Negative Tweets')

plt.show()
