# -*- coding: utf-8 -*-

###############################################################################################################################
# Genera el dataset de las noticias de La Nación para construir el clasificador
###############################################################################################################################

import re
from sklearn.datasets import load_files
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import Article
import pandas as pd
import numpy as np

stemmer = SnowballStemmer('spanish')


def stem_tokens(tokens, stem):
    stemmed = []
    for item in tokens:
        stemmed.append(stem.stem(item))
    return stemmed


def tokenize(text):
    tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')
    tokens = tokenizer.tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


newspaper_data = load_files(r'la_nacion\data', load_content=True, shuffle=True, encoding='utf8')
X, y = newspaper_data.data, newspaper_data.target

documents = []
y_del = []
spanish_stops = set(stopwords.words('spanish'))
spanish_stops.update(['.', ',', '"', "'", '?', '¿', '!', '¡', ':', ';', '(', ')', '[', ']', '{', '}'])  # por las dudas
spanish_stops.update(['algun', 'alguna', 'algunas', 'alguno', 'algunos', 'ambos', 'ampleamos', 'ante', 'antes', 'aquel', 'aquellas', 'aquellos', 'aqui', 'alla', 'alli',
                      'arriba', 'atras', 'bajo', 'bastante', 'bien', 'cada', 'cierta', 'ciertas', 'cierto', 'ciertos', 'como', 'con', 'conseguimos', 'conseguir', 'consigo',
                      'consigue', 'consiguen', 'consigues', 'cual', 'cuando', 'de', 'dentro', 'desde', 'donde', 'dos', 'el', 'ellas', 'ellos', 'empleais', 'emplean', 'emplear',
                      'empleas', 'empleo', 'en', 'encima', 'entonces', 'entre', 'era', 'eramos', 'eran', 'eras', 'eres', 'es', 'ese', 'esta', 'estaba', 'estado', 'estais',
                      'estamos', 'estan', 'esto', 'estoy', 'fin', 'fue', 'fueron', 'fui', 'fuimos', 'gueno', 'ha', 'hace', 'haceis', 'hacemos', 'hacen', 'hacer', 'haces',
                      'hago', 'incluso', 'intenta', 'intentais', 'intentamos', 'intentan', 'intentar', 'intentas', 'intento', 'ir', 'la', 'largo', 'las', 'le', 'les', 'lo',
                      'los', 'mientras', 'mio', 'modo', 'muchos', 'muy', 'ni', 'nos', 'nosotros', 'otro', 'para', 'pero', 'podeis', 'podemos', 'poder', 'podria', 'podriais',
                      'podriamos', 'podrian', 'podrias', 'por', 'por que', 'porque', 'primero', 'puede', 'pueden', 'puedo', 'que', 'quien', 'sabe', 'sabeis', 'sabemos', 'saben',
                      'saber', 'sabes', 'ser', 'si', 'siendo', 'sin', 'sobre', 'sois', 'solamente', 'solo', 'somos', 'soy', 'su', 'sus', 'tambien', 'teneis', 'tenemos', 'tener',
                      'tengo', 'tiempo', 'tiene', 'tienen', 'todo', 'trabaja', 'trabajais', 'trabajamos', 'trabajan', 'trabajar', 'trabajas', 'trabajo', 'tras', 'tuyo', 'ultimo',
                      'un', 'una', 'unas', 'uno', 'unos', 'usa', 'usais', 'usamos', 'usan', 'usar', 'usas', 'uso', 'va', 'vais', 'valor', 'vamos', 'van', 'vaya', 'verdad',
                      'verdadera', 'verdadero', 'vosotras', 'vosotros', 'voy', 'yo'])  # sin tíldes
spanish_stops.update(['estabamos', 'estara', 'estaran', 'estaras', 'estare', 'estareis', 'estaria', 'estariais', 'estariamos', 'estarian', 'estarias', 'esteis', 'esten', 'estes',
                      'estuvieramos', 'estuviesemos', 'fueramos', 'fuesemos', 'habeis', 'habia', 'habiais', 'habiamos', 'habian', 'habias', 'habra', 'habran', 'habras', 'habre',
                      'habreis', 'habria', 'habriais', 'habriamos', 'habrian', 'habrias', 'hayais', 'hubieramos', 'hubiesemos', 'mas', 'mia', 'mias', 'mios', 'seais', 'sera',
                      'seran', 'seras', 'sere', 'sereis', 'seria', 'seriais', 'seriamos', 'serian', 'serias', 'tendra', 'tendran', 'tendras', 'tendre', 'tendreis', 'tendria',
                      'tendriais', 'tendriamos', 'tendrian', 'tendrias', 'tengais', 'tenia', 'teniais', 'teniamos', 'tenian', 'tenias', 'tuvieramos', 'tuviesemos'])  # algunas más...
# Se normalizan las stop words
spanish_stops = stem_tokens(spanish_stops, stemmer)

for docu in range(0, len(X)):
    # extrae el texto del html
    article = Article('', language='es')
    # obtiene el cuerpo de la noticia, luego de la capitalización inicial y hasta el ¿Te gustó esta nota?
    article.set_html(X[docu].split('<p class="capital">', 1)[-1].split('<span class="texto-like">', 1)[0])
    article.parse()
    document = article.text
    # Existen artículos que no comienzan con letra capital, por ejemplo:
    # https://www.lanacion.com.ar/politica/la-justicia-modo-avion-despues-elecciones-nid2284350
    if not document:
        y_del.append(docu)  # guardo el índice para eliminar la clase
        continue  # descarto el documento
    # remueve los dígitos y los dígitos entre palabras
    document = re.sub(r'\w*\d\w*', ' ', document)
    # sustituye nueva línea por un espacio
    document = re.sub(r'\n', ' ', document)
    # sustituye los múltiples espacios por sólo un espacio
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    documents.append(document)

y_new = np.delete(y, y_del)

vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2), min_df=5, max_df=0.8, stop_words=spanish_stops, lowercase=True, strip_accents='unicode')
X = pd.DataFrame(vectorizer.fit_transform(documents).toarray(), columns=vectorizer.get_feature_names())

# exporta el dataset para clasificar
dataset = X.merge(pd.DataFrame(y_new, columns=['target']), left_index=True, right_index=True)
dataset['target_y'] = dataset['target_y'].map({0: 'Economia', 1: 'Politica', 2: 'Turismo'})
dataset.to_pickle(r'la_nacion\datasets\la_nacion_dataset.pkl')
