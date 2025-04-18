import re
from src.setup_nlp import nlp, word_tokenize


def text_preprocessing(text):
    text = text.lower()

    text = re.sub(r'@\w+|#[\w-]+|http\S+|\n', '', text)

    text = re.sub(r'[^\w\s]', ' ', text)

    words = word_tokenize(text, language='english')

    words = [word for word in words if not re.match(r'^_+$', word)]

    processed_words = []

    for word in words:
        try:
            word = nlp(word)[0].lemma_
        except:
            pass

        processed_words.append(word)

    return ' '.join(processed_words)


def tokenize_text(example, tokenizer):
  return tokenizer(example['text'], padding='max_length', truncation=True)


def df_proc(df, tokenizer):
  df = df.map(lambda x: tokenize_text(x, tokenizer))
  df = df.remove_columns(['text', '__index_level_0__'])
  df = df.rename_column('label', 'labels')
  df.set_format('torch')
  return df