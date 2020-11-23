import numpy as np
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer

QUERY_IDX = 5


# Read documents and query
def read_file():
    f = open("input.txt", 'r', encoding='UTF8')
    d = list()
    while True:
        line = f.readline()
        if not line:
            break

        # Convert to lower-cases
        line = line.lower()
        # Replace all non-alphanumeric characters with spaces
        line = re.sub(r'[^a-zA-Z0-9\s]', ' ', line)
        d.append(line)

    f.close()
    q = list()
    q.append(d[QUERY_IDX])
    d.pop()

    return d, q


# Calculate length of documents and query
def calculate_length(t):
    sum = 0
    for i in range(len(t)):
        sum += math.pow(t[i], 2)

    return math.sqrt(sum)


# Calculate cosine similarity values
def calculate_similarity(table, query):
    q = query.ravel()
    query_len = calculate_length(q)
    cos_sim = list()

    i = 1
    for t in table:
        t = t.ravel()
        doc_len = calculate_length(t)
        cos_sim.append([i, np.dot(t, q) / doc_len * query_len])
        i += 1

    return cos_sim


# Calculate the rank order of the documents that match the following query
def rank_order(cos_sim):
    rank = sorted(cos_sim, key=lambda sim: sim[1], reverse=True)

    print('Ranked list:')
    i = 1
    for r in rank:
        print('Rank', i, '= Doc', r[0], '& Similarity score=', r[1])
        i += 1


# Read documents and query
doc, query = read_file()

# Using TfidfVectorizer, fit and transform documents and query
tfidfv = TfidfVectorizer(stop_words='english').fit(doc)
tfidf_table = tfidfv.transform(doc).toarray()
tfidf_query = tfidfv.transform(query).toarray()
cos_sim = calculate_similarity(tfidf_table, tfidf_query)

print('TF X IDF')
print(tfidf_table, '\n')
rank_order(cos_sim)
