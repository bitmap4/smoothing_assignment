from .base import Base, extract_ngrams
import numpy as np
from typing import List, Dict, Tuple

class InterpolationModel(Base):
    def __init__(self, n: int):
        super().__init__(n)

    def train(self, sentences):
        super().train(sentences)
        self.lambdas = self.tune_lambdas()

    def get_probabilities(self, sentence: List[str]) -> Dict[Tuple[str], float]:
        """
        Interpolation smoothing for n-grams.

        Args:
            sentence: List of tokens in the sentence

        Returns:
            Dictionary of n-gram probabilities
        """
        processed_sent = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        ngrams = extract_ngrams(processed_sent, self.n)
        prob_dict = {}
        
        # Validate lambdas
        if not self.lambdas or len(self.lambdas) != self.n or abs(sum(self.lambdas)-1) > 1e-6:
            raise ValueError("Invalid self.lambdas. Ensure they sum to 1.")
        
        for ngram in ngrams:
            total_prob = 0.0
            for k in range(1, self.n + 1):
                kgram = ngram[-k:]  # Get k-gram (last k elements)
                lam = self.lambdas[k-1]
                
                if k == 1:
                    # Laplace smoothing for unigrams: (count + 1) / (total + V)
                    count = self.level_counts[1].get(kgram, 0)
                    total = self.total_counts[1]
                    # prob = (count + 1) / (total + self.V) if total != 0 else 1 / self.V
                    prob = (count / total if total != 0 else 1e-8) or 1e-8
                else:
                    # Laplace smoothing for higher n-grams
                    context = kgram[:-1]
                    # count_context = self.level_counts[k-1].get(context, 0)
                    count_context = self.total_counts[k-1]
                    count_kgram = self.level_counts[k].get(kgram, 0)
                    
                    # Apply add-one smoothing: (count_kgram + 1) / (count_context + V)
                    # prob = (count_kgram + 1) / (count_context + self.V) if count_context != 0 else 1 / self.V
                    prob = (count_kgram / count_context if count_context != 0 else 1e-8) or 1e-8
                
                total_prob += lam * prob
            
            prob_dict[ngram] = total_prob
        
        return prob_dict
    
    def tune_lambdas(self):
        """
        Tune lambdas using deleted interpolation method.
        For each n-gram (w1...wn), compare estimates:
        - f(w1...wn-1,wn)/(f(w1...wn-1))
        - f(w2...wn)/(f(w2...wn-1))
        ...
        - f(wn)/N
        """
        # Initialize lambdas
        lambdas = [0.0] * self.n
        
        # For each n-gram in training data
        for ngram, count in self.ngram_counts.items():
            if count <= 1:  # Skip hapax legomena
                continue
                
            # Calculate estimates for each level
            estimates = []
            for level in range(1, self.n + 1):
                # Get relevant substring of n-gram
                rel_ngram = ngram[-(level):]
                rel_context = ngram[-(level):-1]
                
                # Handle context counts
                if level == 1:
                    context_count = self.total_counts[1]
                else:
                    context_count = self.level_counts[level-1][rel_context]
                
                # Skip if context appeared only once
                if context_count <= 1:
                    estimates.append(0.0)
                    continue
                    
                # Calculate estimate
                ngram_count = self.level_counts[level][rel_ngram]
                estimate = (ngram_count - 1) / (context_count - 1)
                estimates.append(estimate)
            
            # Increment lambda for level with highest estimate
            max_idx = np.argmax(estimates)
            lambdas[max_idx] += count
        
        # Normalize lambdas
        total = sum(lambdas)
        if total > 0:
            lambdas = [l/total for l in lambdas]
        else:
            # Fallback to uniform weights
            lambdas = [1.0/self.n] * self.n
            
        return lambdas