#%% imports
import pandas as pd
import numpy as np
import nltk as nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import *
from nltk.corpus import *
from sklearn.model_selection import train_test_split

#%% read data
data = pd.read_csv('dataset/train.csv')

print(data.head())

#%% map last col
category_mapping = {'Politics': 0, 'Sports': 1, 'Media': 2, 'Market & Economy': 3, 'STEM': 4}
data['Category'] = data['Category'].replace(category_mapping)
print(data['Category'].head())

#%%
def word_tokens(discussion):
  discussion_tokens=[]
  for d in discussion:
    discussion_tokens.append(word_tokenize(d))
  return discussion_tokens

#%%
def drop_1st_col(df):
    df.drop(columns=['SampleID'], inplace=True)
    return df

data = drop_1st_col(data)


#%% clean dataset

def drop_nulls(df):
    print(df.isnull().sum())
    df.dropna(subset=['Discussion'], inplace=True)
    print(df.isnull().sum())
    return df
data = drop_nulls(data)

def to_lowercase(df):
    df['Discussion'] = df['Discussion'].str.lower()
    return df
data = to_lowercase(data)

def count_duplicates_by_columns(df, columns):
    num_duplicates = df.duplicated(subset=columns).sum()
    print(f'num_duplicates = {num_duplicates}, columns = {columns}')

def remove_duplicates(df):
    df = df.drop_duplicates(subset=['Discussion', 'Category'])
    df = df[df['Discussion'].map(df['Discussion'].value_counts()) == 1]
    return df
data = remove_duplicates(data)
count_duplicates_by_columns(data, ['Discussion', 'Category'])

def clean(df):
    # urls
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df = df.applymap(lambda x: re.sub(url_pattern, '', str(x)) if isinstance(x, str) else x)

    # special_chars
    df['Discussion'] = df['Discussion'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    return df

def clean_stop_words(df, column_name):
    # Prepare the stop words set
    stop_words = {word.lower() for word in stopwords.words("english")}

    # Define a function to process individual text
    def remove_stop_words(text):
        tokens = word_tokenize(text)
        return ' '.join([token for token in tokens if token.lower() not in stop_words])

    # Apply the function to the specified column
    df[column_name] = df[column_name].apply(remove_stop_words)
    return df
data = clean(data)
data = clean_stop_words(data, 'Discussion')


# Function to lemmatize text in a DataFrame column
def lemmatize_column(df, column_name):
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
        tokens = word_tokenize(text)
        return ' '.join([lemmatizer.lemmatize(token.lower()) for token in tokens])

    df[column_name] = df[column_name].apply(lemmatize_text)
    return df
# data = lemmatize_column(data, 'Discussion')

def split_data(df):
    X_train, X_val, Y_train, Y_val = train_test_split(df['Discussion'], df['Category'], test_size=0.2, random_state=42, stratify=df['Category'])
    return X_train, X_val, Y_train, Y_val

X_train, X_val, Y_train, Y_val = split_data(data)


