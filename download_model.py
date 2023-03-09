import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk_word_tokens = nltk.download('punkt', download_dir='./nltk_data')
