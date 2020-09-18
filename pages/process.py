import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
# Netflix Recommender Using K Nearest Neighbors

### import packages


```python
import pandas as pd
import tqdm
from tqdm import tqdm

import spacy
```

### Data Exploration

My goal is to create a simple Machine Learning Model that enables the user to input their desired movie description and recieve a list of movie recommendations. The first step of this process is to use a dataset that includes Movie Titles and their respective descriptions. Luckily, Netflix has provided this dataset.


```python
df = pd.read_csv('netflix_titles.csv')
df.head()
```

![](assets/dataframe.png), width=50

```python
df.shape
```




    (6234, 12)




```python
df['type'].value_counts()
```




    Movie      4265
    TV Show    1969
    Name: type, dtype: int64




```python
df['country'].value_counts()
```




    United States                         2032
    India                                  777
    United Kingdom                         348
    Japan                                  176
    Canada                                 141
                                          ... 
    United Kingdom, Pakistan                 1
    Iran, France                             1
    South Africa, China, United States       1
    Lebanon, Jordan                          1
    Spain, Cuba                              1
    Name: country, Length: 554, dtype: int64




```python
df = df[(df['type'] == 'Movie') & (df['country'] == 'United States')]
df = df.reset_index(drop=True)
df.shape
```




    (1482, 12)




```python
df.isnull().sum()
```




    show_id           0
    type              0
    title             0
    director         40
    cast            145
    country           0
    date_added        0
    release_year      0
    rating            3
    duration          0
    listed_in         0
    description       0
    dtype: int64



The Netflix dataset includes an assortment of Movies and TV-Shows from many Countries. I wanted to target only Movies in the United States, so the dataset reduces from 6234 to 1482 obervations.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80125979</td>
      <td>Movie</td>
      <td>#realityhigh</td>
      <td>Fernando Lebrija</td>
      <td>Nesta Cooper, Kate Walsh, John Michael Higgins...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2017</td>
      <td>TV-14</td>
      <td>99 min</td>
      <td>Comedies</td>
      <td>When nerdy high schooler Dani finally attracts...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80060297</td>
      <td>Movie</td>
      <td>Manhattan Romance</td>
      <td>Tom O'Brien</td>
      <td>Tom O'Brien, Katherine Waterston, Caitlin Fitz...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2014</td>
      <td>TV-14</td>
      <td>98 min</td>
      <td>Comedies, Independent Movies, Romantic Movies</td>
      <td>A filmmaker working on a documentary about lov...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70304988</td>
      <td>Movie</td>
      <td>Stonehearst Asylum</td>
      <td>Brad Anderson</td>
      <td>Kate Beckinsale, Jim Sturgess, David Thewlis, ...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2014</td>
      <td>PG-13</td>
      <td>113 min</td>
      <td>Horror Movies, Thrillers</td>
      <td>In 1899, a young doctor arrives at an asylum f...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80057700</td>
      <td>Movie</td>
      <td>The Runner</td>
      <td>Austin Stark</td>
      <td>Nicolas Cage, Sarah Paulson, Connie Nielsen, W...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2015</td>
      <td>R</td>
      <td>90 min</td>
      <td>Dramas, Independent Movies</td>
      <td>A New Orleans politician finds his idealistic ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80045922</td>
      <td>Movie</td>
      <td>6 Years</td>
      <td>Hannah Fidell</td>
      <td>Taissa Farmiga, Ben Rosenfield, Lindsay Burdge...</td>
      <td>United States</td>
      <td>September 8, 2015</td>
      <td>2015</td>
      <td>NR</td>
      <td>80 min</td>
      <td>Dramas, Independent Movies, Romantic Movies</td>
      <td>As a volatile young couple who have been toget...</td>
    </tr>
  </tbody>
</table>
</div>



### Testing out Spacy on 1 Review


```python
df['description'][0]
```




    'When nerdy high schooler Dani finally attracts the interest of her longtime crush, she lands in the cross hairs of his ex, a social media celebrity.'




```python
doc = nlp(df['description'][0])
doc
```




    When nerdy high schooler Dani finally attracts the interest of her longtime crush, she lands in the cross hairs of his ex, a social media celebrity.




```python
doc = [token.lemma_ for token in doc if (token.is_stop == False) and (token.is_punct == False)]
doc
```




    ['nerdy',
     'high',
     'schooler',
     'Dani',
     'finally',
     'attract',
     'interest',
     'longtime',
     'crush',
     'land',
     'cross',
     'hair',
     'ex',
     'social',
     'medium',
     'celebrity']




```python
list1 = pd.Series([" 1", "2", ' 3'])
list1 = list1.apply(lambda x:x.strip())
list1[2]
```




    '3'



### Tokenize, Remove Stop words, and Punct

Before vectorizing the Movies descriptions, I had to apply two preprocessing steps to lowercase and remove whitespace. I also combined the 'listed_in' and 'description' columns. Then I used Spacy's pretrained NLP model (en_core_web_lg) to lemmatize, remove stop words, and punctuations.


```python
df['description']=df['description'].apply(lambda x:x.lower())
df['listed_in']=df['listed_in'].apply(lambda x:x.lower())

df['description']=df['description'].apply(lambda x:x.strip())
df['listed_in']=df['listed_in'].apply(lambda x:x.strip())

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80125979</td>
      <td>Movie</td>
      <td>#realityhigh</td>
      <td>Fernando Lebrija</td>
      <td>Nesta Cooper, Kate Walsh, John Michael Higgins...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2017</td>
      <td>TV-14</td>
      <td>99 min</td>
      <td>comedies</td>
      <td>when nerdy high schooler dani finally attracts...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80060297</td>
      <td>Movie</td>
      <td>Manhattan Romance</td>
      <td>Tom O'Brien</td>
      <td>Tom O'Brien, Katherine Waterston, Caitlin Fitz...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2014</td>
      <td>TV-14</td>
      <td>98 min</td>
      <td>comedies, independent movies, romantic movies</td>
      <td>a filmmaker working on a documentary about lov...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70304988</td>
      <td>Movie</td>
      <td>Stonehearst Asylum</td>
      <td>Brad Anderson</td>
      <td>Kate Beckinsale, Jim Sturgess, David Thewlis, ...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2014</td>
      <td>PG-13</td>
      <td>113 min</td>
      <td>horror movies, thrillers</td>
      <td>in 1899, a young doctor arrives at an asylum f...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80057700</td>
      <td>Movie</td>
      <td>The Runner</td>
      <td>Austin Stark</td>
      <td>Nicolas Cage, Sarah Paulson, Connie Nielsen, W...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2015</td>
      <td>R</td>
      <td>90 min</td>
      <td>dramas, independent movies</td>
      <td>a new orleans politician finds his idealistic ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80045922</td>
      <td>Movie</td>
      <td>6 Years</td>
      <td>Hannah Fidell</td>
      <td>Taissa Farmiga, Ben Rosenfield, Lindsay Burdge...</td>
      <td>United States</td>
      <td>September 8, 2015</td>
      <td>2015</td>
      <td>NR</td>
      <td>80 min</td>
      <td>dramas, independent movies, romantic movies</td>
      <td>as a volatile young couple who have been toget...</td>
    </tr>
  </tbody>
</table>
</div>




```python
nlp = spacy.load("en_core_web_lg")
```


```python
#Extract tokens from reviews using Spacy
# nlp.Defaults.stop_words |= {"my_new_stopword1","my_new_stopword2",}

other_words = ['movie','find',' movie', 'movie ', ' movie ', 'movies']

tokens = []

for index,row in tqdm(df.iterrows()):
    description = row['description']
    listed_in = row['listed_in']
    doc = description + listed_in
    doc = nlp(doc)
    doc = [token.lemma_ for token in doc if (token.is_stop != True) and (token.is_punct != True) and (str(token) not in other_words)]
    tokens.append(doc)
```

    1482it [00:15, 96.07it/s]



```python
tokens[0]
```




    ['nerdy',
     'high',
     'schooler',
     'dani',
     'finally',
     'attract',
     'interest',
     'longtime',
     'crush',
     'land',
     'cross',
     'hair',
     'ex',
     'social',
     'medium',
     'celebrity.comedie']




```python
df['tokens'] = tokens
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
      <th>tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80125979</td>
      <td>Movie</td>
      <td>#realityhigh</td>
      <td>Fernando Lebrija</td>
      <td>Nesta Cooper, Kate Walsh, John Michael Higgins...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2017</td>
      <td>TV-14</td>
      <td>99 min</td>
      <td>comedies</td>
      <td>when nerdy high schooler dani finally attracts...</td>
      <td>[nerdy, high, schooler, dani, finally, attract...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80060297</td>
      <td>Movie</td>
      <td>Manhattan Romance</td>
      <td>Tom O'Brien</td>
      <td>Tom O'Brien, Katherine Waterston, Caitlin Fitz...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2014</td>
      <td>TV-14</td>
      <td>98 min</td>
      <td>comedies, independent movies, romantic movies</td>
      <td>a filmmaker working on a documentary about lov...</td>
      <td>[filmmaker, work, documentary, love, modern, m...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70304988</td>
      <td>Movie</td>
      <td>Stonehearst Asylum</td>
      <td>Brad Anderson</td>
      <td>Kate Beckinsale, Jim Sturgess, David Thewlis, ...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2014</td>
      <td>PG-13</td>
      <td>113 min</td>
      <td>horror movies, thrillers</td>
      <td>in 1899, a young doctor arrives at an asylum f...</td>
      <td>[1899, young, doctor, arrive, asylum, apprenti...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80057700</td>
      <td>Movie</td>
      <td>The Runner</td>
      <td>Austin Stark</td>
      <td>Nicolas Cage, Sarah Paulson, Connie Nielsen, W...</td>
      <td>United States</td>
      <td>September 8, 2017</td>
      <td>2015</td>
      <td>R</td>
      <td>90 min</td>
      <td>dramas, independent movies</td>
      <td>a new orleans politician finds his idealistic ...</td>
      <td>[new, orlean, politician, find, idealistic, pl...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80045922</td>
      <td>Movie</td>
      <td>6 Years</td>
      <td>Hannah Fidell</td>
      <td>Taissa Farmiga, Ben Rosenfield, Lindsay Burdge...</td>
      <td>United States</td>
      <td>September 8, 2015</td>
      <td>2015</td>
      <td>NR</td>
      <td>80 min</td>
      <td>dramas, independent movies, romantic movies</td>
      <td>as a volatile young couple who have been toget...</td>
      <td>[volatile, young, couple, year, approach, coll...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Object from Base Python
from collections import Counter

# The object `Counter` takes an iterable, but you can instaniate an empty one and update it. 
word_counts = Counter()
```


```python
def count(docs):

        word_counts = Counter()
        appears_in = Counter()
        
        total_docs = len(docs)

        for doc in docs:
            word_counts.update(doc)
            appears_in.update(set(doc))

        temp = zip(word_counts.keys(), word_counts.values())
        
        wc = pd.DataFrame(temp, columns = ['word', 'count'])

        wc['rank'] = wc['count'].rank(method='first', ascending=False)
        total = wc['count'].sum()

        wc['pct_total'] = wc['count'].apply(lambda x: x / total)
        
        wc = wc.sort_values(by='rank')
        wc['cul_pct_total'] = wc['pct_total'].cumsum()

        t2 = zip(appears_in.keys(), appears_in.values())
        ac = pd.DataFrame(t2, columns=['word', 'appears_in'])
        wc = ac.merge(wc, on='word')

        wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)
        
        return wc.sort_values(by='rank')
```


```python
wc = count(df['tokens'])
```


```python
import squarify
import matplotlib.pyplot as plt
import seaborn as sns

wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()
```


![png](output_27_0.png)


Here I plotted the top 20 most frequent words in the corpus.

### Vectorize Tokens (TFIDF)

Now that the new Movie descriptions are clean and tokenized in another column called 'tokens'. we are ready to vectorize. In my TF-IDF vectorizer, I included the following hyperparameters:
>  n_gram_range=(1,2): Helps perserve the context of the movie description with 2 linked words

>  max_df=0.97 and min_df=2: Helps trim down my vocab to words that appear in at most 97% and at least 2 of the documents

>  max_features=1000: Best 3000 words/2-grams combinations


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1,2),
                      max_df=0.97,
                      min_df=2,
                      max_features = 3000)

description = df['tokens'].astype(str)

dtm = tfidf.fit_transform(description)

dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())

dtm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>000</th>
      <th>000 year</th>
      <th>10</th>
      <th>10 year</th>
      <th>100</th>
      <th>11</th>
      <th>11 year</th>
      <th>12</th>
      <th>12 year</th>
      <th>13</th>
      <th>...</th>
      <th>youth</th>
      <th>youtube</th>
      <th>youtube sensation</th>
      <th>yuppie</th>
      <th>zach</th>
      <th>zach galifianakis</th>
      <th>zack</th>
      <th>zion</th>
      <th>zombie</th>
      <th>zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3000 columns</p>
</div>



### K-NearestNeighbor


```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
nn.fit(dtm)
```




    NearestNeighbors(algorithm='kd_tree', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     radius=1.0)




```python
nn.kneighbors([dtm.iloc[0].values])
```




    (array([[0.        , 1.23238215, 1.23627091, 1.24601841, 1.26335832]]),
     array([[   0,  751, 1150,  899,  883]]))




```python
nn.kneighbors([dtm.iloc[0]])
```




    (array([[0.        , 1.23238215, 1.23627091, 1.24601841, 1.26335832]]),
     array([[   0,  751, 1150,  899,  883]]))



### Query Movie Recommender


```python
movie = ["Four teenagers are sucked into a magical video game, and the only way they can escape is to work together to finish the game."]
```


```python
new = tfidf.transform(movie)
```


```python
new
```




    <1x3000 sparse matrix of type '<class 'numpy.float64'>'
      with 8 stored elements in Compressed Sparse Row format>




```python
nn.kneighbors(new.todense())
```




    (array([[1.06911197, 1.16598473, 1.19431197, 1.20707636, 1.22899886]]),
     array([[ 704,  238, 1425, 1186, 1454]]))




```python
recommendations = nn.kneighbors(new.todense())[1].tolist()[0]
recommendations
```




    [704, 238, 1425, 1186, 1454]




```python
df[['type','title','cast','listed_in','description']].iloc[recommendations].to_markdown()
```




    "|      | type   | title                                       | cast                                                                                                                                                                                                                                                                                                                                                                  | listed_in                                      | description                                                                                                                                            |\n|-----:|:-------|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|\n|  704 | Movie  | Ralph Breaks the Internet: Wreck-It Ralph 2 | John C. Reilly, Sarah Silverman, Taraji P. Henson, Gal Gadot, Jack McBrayer, Jane Lynch, Alan Tudyk, Ed O'Neill, Susan Lucci, Jason Lee, Idina Menzel, Anika Noni Rose, Judy Reyes, Grant Show, Ana Ortiz, Ming-Na Wen, Rebecca Wisocky, Paige O'Hara, Linda Larkin, Mariana Klaveno, Daria Ramirez, Tom Irwin, Edy Ganem, Irene Bedard, Jodi Benson, Anthony Daniels | children & family movies, comedies             | when video-game bad guy ralph and best friend vanellope discover a way onto the internet, they set off on a mission to save her broken game.           |\n|  238 | Movie  | Barbie: Video Game Hero                     | Erica Lindbeck, Sienna Bohn, Shannon Chan-Kent, Michael Dobson, Alyssya Swales, Rebekah Asselstine, Brad Swaile, Sam Vincent, Ingrid Nilson, Nesta Cooper                                                                                                                                                                                                             | children & family movies                       | pulled into her favorite video game, barbie becomes a fun, roller-skating heroine who's battling a sinister emoji that's trying to take over.          |\n| 1425 | Movie  | Bigger Fatter Liar                          | Ricky Garcia, Jodelle Ferland, Barry Bostwick, Fiona Vroom, Kevin O'Grady, Karen Holness                                                                                                                                                                                                                                                                              | children & family movies, comedies             | when his video game concept is stolen by a dishonest executive, a teenage chronic liar sets out to prove that he's telling the truth, for once.        |\n| 1186 | Movie  | Black Mirror: Bandersnatch                  | Fionn Whitehead, Will Poulter, Craig Parkinson, Alice Lowe, Asim Chaudhry                                                                                                                                                                                                                                                                                             | dramas, international movies, sci-fi & fantasy | in 1984, a young programmer begins to question reality as he adapts a dark fantasy novel into a video game. a mind-bending tale with multiple endings. |\n| 1454 | Movie  | Spy Kids 3: Game Over                       | Daryl Sabara, Sylvester Stallone, Ricardo Montalban, Alexa PenaVega                                                                                                                                                                                                                                                                                                   | children & family movies, comedies             | carmen gets caught in a virtual reality game designed by the kids' new nemesis, the toymaker, and it's up to juni to save her by beating the game.     |"



### Export model


```python
import pickle

filename = 'vect_01.pkl'
pickle.dump(tfidf, open(filename, 'wb'))

filename = 'knn_01.pkl'
pickle.dump(nn, open(filename, 'wb'))
```

            """
            
        ),

    ],
)

layout = dbc.Row([column1])