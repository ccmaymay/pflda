#!/usr/bin/env python


import pylowl
import random


class LdaModel(object):
    def __init__(self, reservoir_size, alpha, beta, num_topics, vocab):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.vocab = vocab
        self.tw_counts = None
        self.dt_counts = None
        self.reservoir = pylowl.ValuedReservoirSampler(reservoir_size)

    def tokenize(self, doc):
        return [vocab[word] for word in doc.split()]

    def sample_topic(self, i, w):
        # TODO
        cmf = self.make_cmf(pmf)

        r = random.random()
        for k in range(len(cmf)):
            if r < cmf[k]:
                return k
        return len(cmf) - 1

    def make_cmf(self, pmf):
        pass # TODO

    def add_doc(self, doc):
        self.reservoir.insert(doc, self.tokenize)

    def learn(self, num_iters):
        sample = self.reservoir.sample()
        assignments = []
        self.tw_counts = [0] * (self.num_topics * vocab.size)
        self.dt_counts = [0] * (len(sample) * self.num_topics)
        for i in range(len(sample)):
            for j in sample[i]:
                w = sample[i][j]
                assignments.append(random.randint(0, self.num_topics - 1))
                self.tw_counts[assignments[-1] * vocab.size + w] += 1
                self.dt_counts[i * self.num_topics + assignments[-1]] += 1
        for t in range(num_iters):
            m = 0
            for i in range(len(sample)):
                for j in sample[i]:
                    w = sample[i][j]
                    self.tw_counts[assignments[m] * vocab.size + w] -= 1
                    self.dt_counts[i * self.num_topics + assignments[m]] -= 1
                    assignments[m] = self.sample_topic(i, w)
                    self.tw_counts[assignments[m] * vocab.size + w] += 1
                    self.dt_counts[i * self.num_topics + assignments[m]] += 1
                    m += 1
            
    
def main():
    pass # TODO


if __name__ == '__main__':
    main(*sys.argv[1:])
