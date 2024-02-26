import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load and preprocess the corpus
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Lemmatize tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
    return corpus

corpus = load_corpus('corpus.txt')

# Tokenize sentences
sent_tokens = nltk.sent_tokenize(corpus)
# Tokenize words
word_tokens = preprocess_text(corpus)

# Keyword matching for greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting_response(text):
    for word in text.split():
        if word.lower() in GREETING_INPUTS:
            return np.random.choice(GREETING_RESPONSES)

# Generating Response
def generate_response(user_input):
    robo_response = ''
    sent_tokens.append(user_input)

    # TF-IDF vectorization
    TfidfVec = TfidfVectorizer(tokenizer=preprocess_text, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    # Calculate cosine similarity
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

# Chat
flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag == True):
    user_input = input()
    user_input = user_input.lower()
    if(user_input != 'bye'):
        if(user_input == 'thanks' or user_input == 'thank you'):
            flag = False
            print("ROBO: You are welcome!")
        else:
            if(greeting_response(user_input) != None):
                print("ROBO: " + greeting_response(user_input))
            else:
                print("ROBO: ", end="")
                print(generate_response(user_input))
                sent_tokens.remove(user_input)
    else:
        flag = False
        print("ROBO: Bye! Take care.")
