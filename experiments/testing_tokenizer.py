
import pickle
from tokenizer2 import SyllableTokenizer

tokenizer_path = 'syllable_tokenizer.pkl'
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

sen1 = "The capital of France is Paris."
print(sen1, "contains", len(tokenizer.tokenize(sen1)), "Tokens like:", tokenizer.tokenize(sen1))
sen2 = "Paris (French pronunciation: [paʁi] ) is the capital and largest city of France."
print(sen2, "contains", len(tokenizer.tokenize(sen2)), "Tokens like:", tokenizer.tokenize(sen2))
sen3 = "The City of Paris is the centre of the Île-de-France region, or Paris Region."
print(sen3, "contains", len(tokenizer.tokenize(sen3)), "Tokens like:", tokenizer.tokenize(sen3))
sen4 = "the population of France. The Paris Region had a"
print(sen4, "contains",  len(tokenizer.tokenize(sen4)), "Tokens like:", tokenizer.tokenize(sen4))
sen5 = "capital"
print(sen5, "contains",  len(tokenizer.tokenize(sen5)), "Tokens like:", tokenizer.tokenize(sen5))
sen6= "France"
print(sen6, "contains",  len(tokenizer.tokenize(sen6)), "Tokens like:", tokenizer.tokenize(sen6))
sen7 = "Paris"
print(sen7, "contains",  len(tokenizer.tokenize(sen7)), "Tokens like:", tokenizer.tokenize(sen7))