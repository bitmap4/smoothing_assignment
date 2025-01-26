from .base import Base, extract_ngrams
from typing import List, Dict, Tuple

class LaplaceModel(Base):
    def get_probabilities(self, sentence: List[str]) -> Dict[Tuple[str], float]:
        """
        Laplace (add-one) smoothing for n-grams.

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
            if self.n == 1:
                # For unigrams, denominator is total unigrams + V
                denominator = self.total_counts[self.n] + self.V
            else:
                # For higher n-grams, get the context count
                context = ngram[:-1]
                count_context = self.level_counts[self.n - 1].get(context, 0)
                denominator = count_context + self.V
            # Calculate Laplace-smoothed probability
            prob = (count_ngram + 1) / denominator
            prob_dict[ngram] = prob
        
        return prob_dict