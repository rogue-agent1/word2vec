#!/usr/bin/env python3
"""word2vec - Word2Vec skip-gram with negative sampling from scratch.

Usage: python word2vec.py [--dim D] [--window W] [--demo]
"""
import sys, math, random, collections

def sigmoid(x):
    if x > 20: return 1.0
    if x < -20: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

class Word2Vec:
    def __init__(self, dim=32, window=2, neg_samples=5, lr=0.025, min_count=1):
        self.dim = dim; self.window = window; self.neg_samples = neg_samples
        self.lr = lr; self.min_count = min_count
        self.W = {}; self.C = {}  # target and context embeddings
        self.vocab = {}; self.idx2word = []
        self.word_freq = []; self.neg_table = []

    def build_vocab(self, corpus):
        counts = collections.Counter()
        for sent in corpus:
            counts.update(sent)
        self.vocab = {}; self.idx2word = []; self.word_freq = []
        for w, c in counts.most_common():
            if c >= self.min_count:
                self.vocab[w] = len(self.idx2word)
                self.idx2word.append(w)
                self.word_freq.append(c)
        n = len(self.vocab)
        # Init embeddings
        for i in range(n):
            self.W[i] = [random.gauss(0, 0.1) for _ in range(self.dim)]
            self.C[i] = [0.0] * self.dim
        # Negative sampling table (unigram^0.75)
        power_freq = [f**0.75 for f in self.word_freq]
        total = sum(power_freq)
        self.neg_table = []
        for i, pf in enumerate(power_freq):
            self.neg_table.extend([i] * max(1, int(pf / total * 10000)))

    def train(self, corpus, epochs=5):
        for epoch in range(epochs):
            total_loss = 0; pairs = 0
            for sent in corpus:
                ids = [self.vocab[w] for w in sent if w in self.vocab]
                for i, target in enumerate(ids):
                    start = max(0, i - self.window)
                    end = min(len(ids), i + self.window + 1)
                    for j in range(start, end):
                        if j == i: continue
                        context = ids[j]
                        loss = self._train_pair(target, context)
                        total_loss += loss; pairs += 1
            if pairs > 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/pairs:.4f}, pairs={pairs}")

    def _train_pair(self, target, context):
        # Positive sample
        dot = sum(self.W[target][d] * self.C[context][d] for d in range(self.dim))
        sig = sigmoid(dot)
        grad = self.lr * (1 - sig)
        loss = -math.log(max(sig, 1e-10))
        for d in range(self.dim):
            g = grad * self.C[context][d]
            self.C[context][d] += grad * self.W[target][d]
            self.W[target][d] += g
        # Negative samples
        for _ in range(self.neg_samples):
            neg = random.choice(self.neg_table)
            if neg == context: continue
            dot = sum(self.W[target][d] * self.C[neg][d] for d in range(self.dim))
            sig = sigmoid(dot)
            grad = self.lr * (-sig)
            loss -= math.log(max(1 - sig, 1e-10))
            for d in range(self.dim):
                g = grad * self.C[neg][d]
                self.C[neg][d] += grad * self.W[target][d]
                self.W[target][d] += g
        return loss

    def most_similar(self, word, topn=5):
        if word not in self.vocab: return []
        wid = self.vocab[word]
        vec = self.W[wid]
        norm = math.sqrt(sum(v*v for v in vec)) or 1
        scores = []
        for i in range(len(self.idx2word)):
            if i == wid: continue
            other = self.W[i]
            onorm = math.sqrt(sum(v*v for v in other)) or 1
            cos = sum(a*b for a,b in zip(vec, other)) / (norm * onorm)
            scores.append((cos, self.idx2word[i]))
        scores.sort(reverse=True)
        return scores[:topn]

    def analogy(self, a, b, c, topn=3):
        """a is to b as c is to ?"""
        if any(w not in self.vocab for w in [a,b,c]): return []
        va, vb, vc = self.W[self.vocab[a]], self.W[self.vocab[b]], self.W[self.vocab[c]]
        target = [vb[d] - va[d] + vc[d] for d in range(self.dim)]
        tnorm = math.sqrt(sum(v*v for v in target)) or 1
        scores = []
        exclude = {self.vocab[w] for w in [a,b,c]}
        for i in range(len(self.idx2word)):
            if i in exclude: continue
            other = self.W[i]
            onorm = math.sqrt(sum(v*v for v in other)) or 1
            cos = sum(a*b for a,b in zip(target, other)) / (tnorm * onorm)
            scores.append((cos, self.idx2word[i]))
        scores.sort(reverse=True)
        return scores[:topn]

def main():
    print("=== Word2Vec (Skip-gram + Negative Sampling) ===\n")
    corpus = [
        "the king sat on the throne".split(),
        "the queen sat on the throne".split(),
        "the king wore a crown".split(),
        "the queen wore a crown".split(),
        "a man walked in the castle".split(),
        "a woman walked in the castle".split(),
        "the prince is the son of the king".split(),
        "the princess is the daughter of the queen".split(),
        "the king and queen ruled the kingdom".split(),
        "the man and woman lived in the village".split(),
    ] * 50  # Repeat for more training
    w2v = Word2Vec(dim=16, window=2, neg_samples=3, lr=0.05)
    w2v.build_vocab(corpus)
    print(f"Vocab: {len(w2v.vocab)} words")
    w2v.train(corpus, epochs=10)
    print(f"\nMost similar to 'king':")
    for score, word in w2v.most_similar("king"):
        print(f"  {word:15s} {score:.4f}")
    print(f"\nMost similar to 'woman':")
    for score, word in w2v.most_similar("woman"):
        print(f"  {word:15s} {score:.4f}")
    print(f"\nAnalogy: king - man + woman = ?")
    for score, word in w2v.analogy("king", "queen", "man"):
        print(f"  {word:15s} {score:.4f}")

if __name__ == "__main__":
    main()
