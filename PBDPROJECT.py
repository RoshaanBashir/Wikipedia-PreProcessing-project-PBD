#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[17]:


#Importing files 
import os
folderpath = r"E:\New folder"  #Path of the file
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
files = [0]  #since im doing only one file so i'll take the first file

for path in filepaths:
    with open(path, 'r', encoding="ISO-8859-1") as f:
        file = f.readlines()
        files.append(file)


# In[18]:


files


# In[19]:


# Evaluating Size of files
sizeOfmyfiles=len(files)
print(sizeOfmyfiles)


# In[20]:


# Coverting to String
listToStr= ' '.join(map(str, files))


# In[21]:


listToStr


# In[22]:


punctuations='''!@#$%^&*()[]{}:;',._~\\'''

my_str="'[\'[[Anarchism]]\\n\', \'\\n\', \'Anarchism is a political philosophy that advocates stateless societies often defined as self-governed voluntary institutions, 'ANARCHISM, a social philosophy that rejects authoritarian government and maintains that voluntary institutions are best suited to express man'\\\'s natural social tendencies. George Woodcock. Anarchism at The Encyclopedia of Philosophy In a society developed on these lines, the voluntary associations which already now begin to cover all the fields of human activity would take a still greater extension so as to substitute themselves for the state in all its functions.Peter Kropotkin.Anarchism from the EncyclopÃ¦dia Britannica Anarchism. The Shorter Routledge Encyclopedia of Philosophy. 2005"

no_punctuation=""
for char in my_str:
    if(char not in punctuations):
        no_punctuation=no_punctuation+char


# In[23]:


no_punctuation


# In[24]:


#Importing Library
import nltk 
nltk.download('punkt')


# In[25]:


nltk.word_tokenize('Anarchismn n Anarchism is a political philosophy that advocates stateless societies often defined as self-governed voluntary institutions ANARCHISM a social philosophy that rejects authoritarian government and maintains that voluntary institutions are best suited to express mans natural social tendencies George Woodcock Anarchism at The Encyclopedia of Philosophy In a society developed on these lines the voluntary associations which already now begin to cover all the fields of human activity would take a still greater extension so as to substitute themselves for the state in all its functionsPeter KropotkinAnarchism from the EncyclopÃ¦dia Britannica Anarchism The Shorter Routledge Encyclopedia of Philosophy 2005')


# In[26]:


#Tokenizing and deleting stopwords
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
allwords=stopwords.words('english')
print(allwords)


# In[27]:


text='Anarchismn n Anarchism is a political philosophy that advocates stateless societies often defined as self-governed voluntary institutions ANARCHISM a social philosophy that rejects authoritarian government and maintains that voluntary institutions are best suited to express mans natural social tendencies George Woodcock Anarchism at The Encyclopedia of Philosophy In a society developed on these lines the voluntary associations which already now begin to cover all the fields of human activity would take a still greater extension so as to substitute themselves for the state in all its functionsPeter KropotkinAnarchism from the EncyclopÃ¦dia Britannica Anarchism The Shorter Routledge Encyclopedia of Philosophy 2005'
tokenized_text=word_tokenize(text)
ftext=[word for word in tokenized_text if not word in allwords]


# In[28]:


print(ftext)


# In[29]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
my_str='Anarchismn n Anarchism is a political philosophy that advocates stateless societies often defined as self-governed voluntary institutions ANARCHISM a social philosophy that rejects authoritarian government and maintains that voluntary institutions are best suited to express mans natural social tendencies George Woodcock Anarchism at The Encyclopedia of Philosophy In a society developed on these lines the voluntary associations which already now begin to cover all the fields of human activity would take a still greater extension so as to substitute themselves for the state in all its functionsPeter KropotkinAnarchism from the EncyclopÃ¦dia Britannica Anarchism The Shorter Routledge Encyclopedia of Philosophy 2005'
my_str=nltk.word_tokenize(my_str)
for word in my_str:
    print(lemmatizer.lemmatize(word))


# In[30]:


# Removing Special Characters
import re
updated=[]
updated=[re.sub('[^a-zA-Z0-9]+','',_) for _ in my_str]
updated #Now File is updated


# In[31]:


#Performing k-means clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import warnings
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


vectorizer = TfidfVectorizer(stop_words='english') #Vactorizing
X = vectorizer.fit_transform(updated)

k = 6 #num of clusters
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),


# In[32]:


#END


# In[ ]:




