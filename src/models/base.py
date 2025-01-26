from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple

class Base:
    def __init__(self, n: int):
        self.n = n
        self.vocab = {'<s>','</s>', '<unk>'}
        self.level_counts = defaultdict(Counter)
        self.total_counts = defaultdict(int)

    def train(self, sentences: List[List[str]]):
        """
        Args:
            sentences: List of tokenized sentences
        """
        self.sentences = sentences
        # Initialize n-gram counts for all levels (1 to n)
        for sent in sentences:
            self.vocab.update(sent)
            sent = ['<s>'] * (self.n - 1) + sent + ['</s>']
            for level in range(1, self.n+1):
                ngrams = extract_ngrams(sent, level)
                self.level_counts[level].update(ngrams)
                self.total_counts[level] += len(ngrams)
        
        self.V = len(self.vocab)
        self.ngram_counts = self.level_counts[self.n]
        # print(self.level_counts[2])
        # print(self.V, self.total_counts[2])
        self.freq_of_freqs = Counter(self.ngram_counts.values())
        

    def get_probabilities(self, sentence: List[str]) -> Dict[Tuple[str], float]:
        """
        No smoothing for n-grams.

        Args:
            sentence: List of tokens in the sentence

        Returns:
            Dictionary of n-gram probabilities
        """
        processed_sent = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        ngrams = extract_ngrams(processed_sent, self.n)
        prob_dict = {}
        
        for ngram in ngrams:
            count_ngram = self.ngram_counts.get(ngram, 0)
            prob = (count_ngram / self.total_counts[self.n] if self.total_counts[self.n] != 0 else 1e-8) or 1e-8
            prob_dict[ngram] = prob
        
        return prob_dict

    def calculate_perplexity(self, prob_dict):
        """
        Calculate perplexity for a test sentence given a probability dictionary
        
        Args:
            prob_dict: Dictionary of n-gram probabilities from smoothing method
            
        Returns:
            Perplexity score (float)
        """
        # Calculate log probability of the test sentence
        log_prob = 0.0
        for ngram, prob in prob_dict.items():
            log_prob += math.log(prob) if prob > 0 else float('-inf')
        
        # Calculate perplexity
        M = len(prob_dict)
        perplexity = math.exp(-log_prob / M) if M > 0 else float('inf')
        return perplexity
    
def extract_ngrams(text: any, n: int):
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams.append(tuple(ngram))
    return ngrams

def sentence_probability(prob_dict):
    return math.exp(sum(math.log(prob) for prob in prob_dict.values()))