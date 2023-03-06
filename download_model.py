import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk_word_tokens = nltk.download('punkt', download_dir='./nltk_data')

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

tokenizer.save_pretrained("./model")
model.save_pretrained("./model")
