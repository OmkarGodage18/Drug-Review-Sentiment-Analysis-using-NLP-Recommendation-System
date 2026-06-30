#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


# In[2]:


#!/usr/bin/env python
# coding: utf-8

# # Data Understanding & Exploration

# In[1]:


import sqlite3

# 1. Connect to the database
conn = sqlite3.connect(r"C:\Users\admin\Downloads\classification.db")
cursor = conn.cursor()

# 2. Query the internal SQLite master table
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# 3. Fetch and print names
tables = cursor.fetchall()

print("Tables in this database:")
for table in tables:
    print(f"- {table[0]}")

# 4. Close connection
conn.close()


# In[2]:


import sqlite3
import pandas as pd

# 1. Establish connection
db_path = r"C:\Users\admin\Downloads\classification.db"
conn = sqlite3.connect(db_path)





        # 3. Read the table into a DataFrame
data = pd.read_sql_query(f"SELECT * FROM drug_reviews_sentiment_analysis", conn)



# finally:
    # 4. Always close the connection
    # conn.close()


# In[3]:


data.head(10)


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data['condition'].value_counts()


# In[7]:


data['condition']=data['condition'].fillna(data['condition'].mode()[0])


# In[8]:




# In[9]:


data.isnull().sum()


# In[10]:


data.describe(include='all')


# In[11]:


data.info()


# In[12]:


data['rating'].value_counts()


# In[13]:


data['review'].value_counts()


# In[14]:


data['drugName'].nunique()


# In[15]:


data['drugName'].unique()


# In[16]:


data['condition'].unique()


# In[17]:


data['condition'].nunique()


# In[18]:


data['condition'].value_counts().head(10)


# In[19]:


data['rating'].describe()


# In[20]:


import matplotlib.pyplot as plt

data['rating'].value_counts().sort_index().plot(kind='bar')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()


# In[21]:


data['review_length'] = data['review'].astype(str).apply(len)


# In[22]:


data['review_length'].plot(kind='hist', bins=100)
plt.title("Review Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()


# # Sentiment Label Creation

# In[23]:

data = data[data['rating'] != 5]

def create_sentiment(rating):

    if rating >= 6:
        return "positive"
    else:
        return "negative"

data['sentiment']=data['rating'].apply(create_sentiment)


# In[24]:


data


# In[25]:


data['sentiment'].value_counts().plot(kind='bar')
plt.title('sentiment class distribution ')
plt.xlabel('sentiment')
plt.ylabel('count')
plt.show()




# # Text Preprocessing

# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import regex as re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# In[29]:


stopwords_list = stopwords.words('english')
exclude= string.punctuation


# In[30]:


stopwords_list


# In[31]:


exclude


# In[32]:


# remove punctuation
def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))


# In[33]:


def remove_stopwords(text):
    data=[word for word in text.split() if word not in stopwords_list]
    return " ".join(data)


# In[34]:

lemmatizer = WordNetLemmatizer()

def apply_lemmatization(text):

    words = [lemmatizer.lemmatize(word)
             for word in text.split()]

    return " ".join(words)

# In[35]:


def text_data_cleaning(data):

    data['review_original'] = data['review'].copy()

    data['review'] = data['review'].fillna('').astype(str)

    data['review'] = data['review'].str.lower()

    data['review'] = data['review'].str.replace(r'&#\d+;',' ',regex=True)

    data['review'] = data['review'].apply(remove_punc)

    data['review'] = data['review'].apply(remove_stopwords)

    data['review'] = data['review'].apply(apply_lemmatization)

    return data


# In[36]:




def tfidf_feature_fit(data):

    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        stop_words='english'
    )

    tfidf_matrix = tfidf.fit_transform(data['review'])

    return tfidf, tfidf_matrix


def tfidf_features_transform(tfidf, data):

    tfidf_matrix = tfidf.transform(data['review'])

    return tfidf_matrix




# In[3]:


# In[37]:


get_ipython().system('jupyter nbconvert --to script data_preprocessing.ipynb')


# In[38]:


# data.to_csv("drug_reviews_sentiment_analysis.csv", index=False)


# In[ ]:


# In[5]:


data.to_csv("drug_reviews_sentiment_analysis.csv", index=False)


# In[ ]:




