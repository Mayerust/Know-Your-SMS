import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download essential NLP resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()  # Initialize the stemmer

def transform_text(text):
    """
    Process raw text: lowercase, tokenize, filter, and stem.
    
    Args:
        text (str): Input message.
    Returns:
        str: Processed text.
    """
    text = text.lower()  # Normalize to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize

    # Retain only alphanumeric tokens
    alphanum_tokens = [token for token in tokens if token.isalnum()]
    
    # Remove stopwords and punctuation tokens
    filtered_tokens = [
        token for token in alphanum_tokens
        if token not in stopwords.words('english') and token not in string.punctuation
    ]
    
    # Apply stemming to the filtered tokens
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]

    return ' '.join(stemmed_tokens)