import math
from collections import Counter
import numpy as np
import itertools
import regex as re
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity


docs=[]
n_doc=int(input())
for i in range(n_doc):
    docs.append(input())
query = input()
n_dim = int(input())
def preprocess_doc(text):
    text = "".join(re.findall(r"[\w\s]",text))
    text = re.sub(r"\s",' ',text).strip()
    text = re.sub(r"\s+",' ',text).lower().split()
    # stoplist=set('i are at my this on for a of the and to in'.split())
    # text = [word for word in text if word not in stoplist]
    # text = [lema_dict.get(word) if word in lema_dict.keys() else word for word in text]
    return text
docs = [preprocess_doc(doc) for doc in docs]
query = preprocess_doc(query)

unique_terms = list(set((itertools.chain(*docs))))
docs_emb=[]
for doc in docs:
    doc_emb=[]
    for unique_term in unique_terms:
        if unique_term in doc:
            doc_emb.append(1)
        else:
            doc_emb.append(0)
    docs_emb.append(np.array(doc_emb))
docs_emb_array = np.array(docs_emb)

query_emb=[]
for unique_term in unique_terms:
    if unique_term in query:
        query_emb.append(1)
    else:
        query_emb.append(0)
query_emb_array = np.array(query_emb)

U, Sigma, VT = svd(docs_emb_array, full_matrices=False)


k = 2
U_k = U[:, :k]
Sigma_k = np.diag(Sigma[:k])
VT_k = VT[:k, :]

query_projection = query_emb_array @ VT_k.T @ np.linalg.inv(Sigma_k)


documents_projection = U_k @ Sigma_k

similarities = cosine_similarity([query_projection], documents_projection)[0]
print([float(round(x,2)) for x in list(similarities)])
