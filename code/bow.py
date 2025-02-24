from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.tokenize import word_tokenize

corpus = [
    "I like to eat apple",
    "You like to watch TV",
]

# bag of words
vectorizer = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
X = vectorizer.fit_transform(corpus)
print("The bag of words is:")
print(X.toarray())
print("The vocabulary is:")
print(vectorizer.get_feature_names_out())

corpus = [
    "I like to eat apple",
    "You like to watch TV",
]
corpus = " ".join(corpus)

# bag of words
bow = Counter(corpus.split())
print(bow)
