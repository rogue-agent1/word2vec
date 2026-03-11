#!/usr/bin/env python3
"""Simple word2vec (skip-gram with negative sampling)."""
import sys, random, math, collections, re
random.seed(42)
text="the king loves the queen and the queen loves the king the prince loves the princess"
words=text.split(); vocab=list(set(words)); w2i={w:i for i,w in enumerate(vocab)}
dim=10; W=[[random.gauss(0,0.1) for _ in range(dim)] for _ in vocab]
C=[[random.gauss(0,0.1) for _ in range(dim)] for _ in vocab]
def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))
def dot(a,b): return sum(ai*bi for ai,bi in zip(a,b))
lr=0.025; window=2
for epoch in range(50):
    loss=0
    for i,w in enumerate(words):
        for j in range(max(0,i-window),min(len(words),i+window+1)):
            if i==j: continue
            wi,wj=w2i[w],w2i[words[j]]
            s=sigmoid(dot(W[wi],C[wj])); loss-=math.log(s+1e-10)
            g=lr*(1-s)
            for d in range(dim): W[wi][d]+=g*C[wj][d]; C[wj][d]+=g*W[wi][d]
            for _ in range(2):
                neg=random.randint(0,len(vocab)-1)
                s=sigmoid(-dot(W[wi],C[neg])); loss-=math.log(s+1e-10)
                g=lr*(s-1)
                for d in range(dim): W[wi][d]-=g*C[neg][d]
def similarity(w1,w2):
    v1,v2=W[w2i[w1]],W[w2i[w2]]
    d=dot(v1,v2)/(math.sqrt(dot(v1,v1))*math.sqrt(dot(v2,v2))+1e-10)
    return d
print("Word similarities:")
for w1,w2 in [("king","queen"),("king","prince"),("loves","the")]:
    print(f"  {w1} ↔ {w2}: {similarity(w1,w2):.4f}")
