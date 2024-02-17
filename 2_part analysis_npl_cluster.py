# Разведочный анализ данных (англ. exploratory data analysis, EDA) — анализ основных свойств данных, нахождение в них общих закономерностей, распределений и аномалий

# In[1]:


import pandas as pd
import io
#specify column names

column_names = ["id", "reestr_number", "FLD_2", "FLD_3", "FLD_4", "FLD_5",
                      "FLD_6", "FLD_7", "FLD_8", "FLD_9", "FLD_10", "FLD_11", "FLD_12",
                      "FLD_13", "FLD_14", "FLD_15", "FLD_16", "FLD_17", "FLD_18", "FLD_19", "FLD_20",
                      "price", "FLD_22", "FLD_23", "FLD_24", "object_name", "object_code"]

dtypes = { 'id': lambda x: pd.to_numeric(x, errors="coerce"),
          
          'reestr_number': str, 
          'object_name': str,
          'object_code': str,
           'price':lambda x: pd.to_numeric(x, errors="coerce")
    
    
    
    
    
}

df = pd.read_csv('fz.csv',names=column_names,error_bad_lines=False, low_memory=False,nrows=1000000, sep = ',')

#df.info(memory_usage='deep')


# In[2]:


type(df)
    


# In[3]:


df.info()


# In[4]:


df.isnull().any().any() # проверка есть ли  в датасете пустые значения


# In[5]:


df.isnull().sum()


# In[6]:


print(df.shape)


# In[7]:


df.loc[1]


# In[8]:


# распределение выборки по классам
class_counts = df.groupby('object_code').size()
print(class_counts)


# In[9]:


df.select_dtypes(include=['float64', 'int64'])


# In[10]:


df.describe()


# In[11]:


df['object_name']  = df['object_name'].astype('string')
df['object_code']  = df['object_code'].astype('string')


# In[12]:


df.dtypes


# In[13]:


df.select_dtypes(include=['float64', 'int64'])


# In[14]:


grouped_data = df.groupby('object_code')['object_code'].unique()
grouped_data


# In[15]:


df['object_code']


# In[16]:


duplicated_registy_numbers = df[df['reestr_number'].duplicated(keep=False)].sort_values(by='reestr_number')
print(duplicated_registy_numbers[['reestr_number','reestr_number']].head(20))

full_duplicates = duplicated_registy_numbers.duplicated(keep=False).sum()
partial_duplicates = len(duplicated_registy_numbers) - full_duplicates

print('full_duplicates:', full_duplicates)
print('part_duplicates:', partial_duplicates)


# Для того, чтобы отфильтровать данные в датафрейме pandas с помощью регулярного выражения, можно воспользоваться функцией contains с указанием параметра regex для строковых типов данных.По условию ТЗ нужно 

# In[17]:


filtered_df = '''object_code.str.contains("41.") or object_code.str.contains("42.") or object_code.str.contains("43.") or object_code == "71.1" '''


# In[18]:


filtered_df


# In[19]:


df = df.query(filtered_df, engine='python')


# In[20]:


df


# In[21]:


df['object_code']


# In[22]:


df.sort_values(ascending=False, by='price')


# In[23]:


code_counts = df.groupby(['object_code', 'object_name']).size().reset_index(name='count')
code_counts


# In[24]:


grouped_data = df.groupby('object_code')['object_name'].nunique()

multiple_descriptions = grouped_data[grouped_data > 1]
multiple_descriptions


# In[25]:


df.duplicated().sum()


# In[26]:


#Коды ОКПД-2 имеют больше одного наименования.

df.sort_values(ascending=False, by='price') # сортируем по цене


# In[27]:


df['price'] = pd.to_numeric(df['price'], errors='coerce')


# In[28]:


df.info()


# In[29]:


low_price = df[df['price'] < 1000].sort_values(by='price')
low_price


# In[30]:


# удаляем  повторы в реестровом номере

df.drop_duplicates(subset=['reestr_number'], inplace=True)
df = df.dropna()


# In[31]:


df.describe().T


# In[32]:


df.to_csv(r'C:\Data\dataset\clean_dataset.csv',index= False)  #  сохраняем в файл результаты


# In[33]:


from pandas import set_option
set_option('display.width',100)
set_option('display.precision',3)
correlations= df.corr(method='pearson')
print(correlations)


# In[34]:


# оценка ассиметрии распредления атрибутов
skew = df.skew ()
print (skew)


# In[35]:


from matplotlib import pyplot
df.hist()
pyplot.show()


# In[36]:


# построение кореляционной матрицы

from pandas import set_option
set_option('display.width',100)
set_option('display.precision',3)
correlations= df.corr(method='pearson')
print(correlations)

correlations= df.corr()
print(correlations)


# In[37]:


#Корреляционный анализ
import matplotlib.pyplot as plt
import seaborn as sns


# In[38]:


df.info()


# In[39]:


correlation = df[['price', 'object_code']].corr()


plt.figure(figsize=(10, 4))
sns.scatterplot(data=df, x='object_code', y='price', alpha=0.8)
plt.title('Зависимость признаков')
plt.yscale('log')
plt.show()

plt.figure(figsize=(6, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', cbar=False)
plt.title('Корреляция признаков')
plt.show()


# Заказчик не предоставил информцаию по датам.
# Построить зависимость  цены от продолжительности контракта на данном этапе не удалось.
# Необходимо обратиться к заказчику для уточнения  описания предоставленных данных.

# In[40]:


#Проанадизуем ОКПД
plt.figure(figsize=(16, 6))
sns.countplot(data=df, x='object_code', order=df['object_code'].value_counts().index)
plt.title('По_каждому_коду_ОКПД_количество_закупок')
plt.xlabel('Код_ОКПД')
plt.ylabel('Количество')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[41]:


# количество закупок по группам 

df['code_int'] = df['object_code'].apply(lambda x: '.'.join(x.split('.')[:1]))

plt.figure(figsize=(15, 6))
sns.countplot(data=df, x='code_int', order=df['code_int'].value_counts().index)
plt.title('По_группам_ОКПД_количество_закупок')
plt.xlabel('Группа_ОКПД')
plt.ylabel('Количество')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[42]:


df.groupby('code_int')['object_name'].nunique()


# Обработка NLP

# In[43]:


# устанавливаем библиотеки
get_ipython().system(' pip install scikit-learn')
get_ipython().system(' pip install nltk')
get_ipython().system(' pip install pymorphy2')
get_ipython().system(' pip install requests')
get_ipython().system(' pip install gensim')
get_ipython().system(' pip install pymystem3')
get_ipython().system(' pip install matplotlib')


# In[44]:


import numpy as np
import pandas as pd


# In[45]:


column_names = ["id", "reestr_number", "FLD_2", "FLD_3", "FLD_4", "FLD_5",
                      "FLD_6", "FLD_7", "FLD_8", "FLD_9", "FLD_10", "FLD_11", "FLD_12",
                      "FLD_13", "FLD_14", "FLD_15", "FLD_16", "FLD_17", "FLD_18", "FLD_19", "FLD_20",
                      "price", "FLD_22", "FLD_23", "FLD_24", "object_name", "object_code"]

dtypes = { 'id': lambda x: pd.to_numeric(x, errors="coerce"),
          
          'reestr_number': str, 
          'object_name': str,
          'object_code': str,
           'price':lambda x: pd.to_numeric(x, errors="coerce")
    
         }
    
    
    


# In[46]:


df = pd.read_csv('clean_dataset.csv',names=column_names,error_bad_lines=False, low_memory=False,nrows=1000000, sep = ',')
print(df.shape)


# In[47]:


import pandas as pd
import numpy as np


# In[48]:


columns = ['FLD_0', 'FLD_1', 'FLD_25', 'FLD_26', 'FLD_21']
dtypes = {'FLD_0': lambda x: pd.to_numeric(x, errors="coerce"), 
          'FLD_1': str, 
          'FLD_25': str,
          'FLD_26': str, 
          'FLD_21':lambda x: pd.to_numeric(x, errors="coerce")} 


# In[49]:


df = pd.read_csv('C:\Data\dataset\clean_dataset.csv', sep=',', low_memory=False, error_bad_lines=False)


# In[50]:


print(df.shape)
df.describe()


# In[51]:


df.info()


# In[52]:


# преобразуем в стрки
df['object_code']  = df['object_code'].astype('string')
df['object_name']  = df['object_name'].astype('string')
df.dtypes


# In[53]:


df.tail()


# In[54]:


df.head()


# In[55]:


grouped_data = df.groupby('object_code')['object_code'].unique()
grouped_data


# Согласно условиям ТЗ определяем нужные коды.
# Прочие
# Строительно-монтажные работы (СМР) - 41, 42, 43(кроме нижеперечисленных)
# Проектно-изыскательские работы (ПИР) - 41.1, 71.1
# Подключение коммуникаций - 43.22
# 

# In[56]:


query = '''object_code.str.contains("41.") or object_code.str.contains("42.") or object_code.str.contains("43.") or object_code == "71.1" '''
df = df.query(query, engine='python')


# In[57]:


df['object_code']


# удаляем латинские числа и символы
# 
# 

# In[58]:


def clear_text(text):
    clean = [char for char in text if 1000 < ord(text[0])]
    return ''.join(clean)


# In[59]:


df['object_name'] = df['object_name'].apply(lambda x: clear_text(x))
df = df[df["object_name"] != ""]


# In[60]:


df.describe().T


# In[61]:


# группируем по столбцу код окпд
grouped_data = df.groupby('object_code')['object_code'].unique()
grouped_data


# In[62]:


filtered_df = '''object_code.str.contains("41.") or object_code.str.contains("42.") or object_code.str.contains("43.") or object_code == "71.1" '''


# In[63]:


df = df.query(filtered_df, engine='python')


# In[64]:


# displaying the DataFrame
display(df)


# In[65]:


df.to_csv(r'C:\Data\dataset\dataset_filtered.csv', sep=',', index=False)


#  токены,
#  лематизизация слов,
#  шум

# In[66]:


#стоп слова русского языка
import nltk
#nltk.download()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# In[67]:


from nltk import download
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
from nltk.corpus import wordnet as wn 
from nltk.corpus import stopwords


# In[68]:




tag_map = defaultdict(lambda : wn.NOUN) # стоп слова
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

stemmer = nltk.stem.snowball.RussianStemmer()
word_lemmatized = WordNetLemmatizer()


# In[69]:


class MyClass():
    def __init__(self,__df):
        self.df =__df

    def tokening(self):
         self.df['data'] =  [word_tokenize(entry.lower()) for entry in self.df['object_name']]

    def lammatize(self, word, tag):
         return word_lemmatized.lemmatize(word, tag_map[tag])

    def steming(self, word_final):
         return stemmer.stem(word_final)

    def fiting(self): 
        self.tokening()
        for index, entry in zip(self.df["data"].index, self.df['data']):
            final_words = []
            for word, tag in pos_tag(entry):
                if word not in stopwords.words("russian") and word.isalpha():
                    word_final = self.lammatize(word, tag[0])
                    word_final = self.steming(word_final)
                    final_words.append(word_final)
            self.df.loc[index, "data_final"] = str(final_words)
        return self.df



# In[70]:



nlp = MyClass(df)
data_final= nlp.fiting()


# In[71]:


df.to_csv(r'C:\Data\dataset\my_dataset_nlp.csv', sep=',', index=False)


# Step 4 - Построение модели классификатора

# In[73]:


get_ipython().system(' pip install scikit-learn')
get_ipython().system(' pip install nltk')
get_ipython().system(' pip install requests')
get_ipython().system(' pip install matplotlib')


# In[78]:


import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')


# In[83]:


df = pd.read_csv('my_dataset_nlp.csv',names=None,error_bad_lines=False, low_memory=False, sep = ',')


# In[84]:


df


# In[104]:


#Делаем разбивку по группам на основании номеров ОКПД-2
import re 

def match(templ, value):
    try:
        return re.search(f'^{templ}.', value) 
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return False
    
def get_group(value):    
    if (match('41', value) or match('42', value) or match('43', value)) and  value not in ('43.9', '42.9', '42.2', '43.2', '41.1', '71.1', '43.2'):  
        return 'Строительно-монтажные работы'
    elif value in ('41.1', '71.1'):  
        return 'Проектно-изыскательские работы'
    elif value in ('43.2'):  
        return 'Подключение коммуникаций'
    elif value in ('43.9', '42.9', '42.2'):  
        return 'Строительный надзор'
    else:
        return 'Прочие'


# In[105]:


df.info()


# In[106]:


df['object_name']  = df['object_name'].astype('string')
df['object_code']  = df['object_code'].astype('string')



# In[107]:



df['group'] = df['object_code'].apply(lambda x: get_group(x))


# In[96]:


df.head()


# In[97]:


#LabelEncoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['group'])
#y = label_encoder.fit_transform(df['object_code'])
y


# In[99]:


#OneHotEncoder
df_final = df.copy()

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()
data_new = onehotencoder.fit_transform(df_final['group'].values.astype('U').reshape(-1, 1))
columns = np.char.strip(onehotencoder.categories_[0].astype('U'))
df_one_hot_codes = pd.DataFrame(data_new.toarray(), columns=columns)
df_one_hot_codes.head(15)


# In[108]:


#TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer(max_features=1700, min_df=5, max_df=0.7)

X = df['data_final']
tfidf_X = tfidf.fit_transform(X).toarray()

tfidf_X
tfidf = pd.DataFrame(tfidf_X)
tfidf['price'] = df['object_code'].values
#tfidf['y'] = y
tfidf = tfidf.join(df_one_hot_codes)
tfidf


# In[109]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(tfidf.values)
X_train_scaled = scaler.transform(tfidf.values)
X_train_scaled


# In[110]:


#Кластеризация MiniBatchKMeans

from sklearn.cluster import MiniBatchKMeans

#X = df_vectors.values
X = X_train_scaled
kmeans = MiniBatchKMeans(n_clusters=4, random_state=0, batch_size=1000, max_iter=10, n_init="auto").fit(X)
kmeans.cluster_centers_


# In[111]:


df_cluster = df[['object_name', 'object_code', 'price', 'FLD_15', 'group']].copy()
df_cluster['cluster'] = kmeans.labels_
df_cluster.head(100)


# In[112]:


df_cluster['cluster'].value_counts()


# In[113]:


df_cluster['cluster'].hist()


# In[114]:


import matplotlib.pyplot as plt

plt.scatter(df_cluster['object_code'], df_cluster['cluster']) 
plt.show()


# In[115]:


clusters = df_cluster['cluster'].unique()

comparisons = {}
for cluster in  clusters:
    codes = df_cluster[df_cluster['cluster']==cluster].groupby(['object_code']).count()['object_name']
    comparisons[cluster] = codes.index

comparisons


# In[117]:


def get_group(x):
    for key, value in comparisons.items():
        #print(x, value)
        if x in value:            
            return key
    return 0


# In[118]:



df_cluster.groupby(['group','cluster', 'object_code'])['object_name'].count() 


# In[119]:


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(16, 6))
sns.countplot(data=df_cluster, x='group', order=df_cluster['group'].value_counts().index)
plt.title('Количество закупок по каждому коду ОКПД')
plt.xlabel('Код ОКПД')
plt.ylabel('Количество')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[124]:


grouped_data_int = df_cluster.groupby('cluster').agg(
    max_price=('price', 'max'),
    avg_price=('price', 'mean'),
    min_price=('price', 'min')).reset_index()

grouped_data_int


# In[ ]:




