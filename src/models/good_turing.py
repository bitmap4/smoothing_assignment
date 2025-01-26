from .base import Base, extract_ngrams
import math
import numpy as np
from typing import List, Dict, Tuple

class GoodTuringModel(Base):
    def get_probabilities(self, sentence: List[str], threshold=5) -> Dict[Tuple[str], float]:
        """
        Good-Turing smoothing with a piecewise approach:
        - For small r (< threshold), keep Nr as is.
        - For large r (>= threshold), estimate Nr via a regression line:
          log(Nr) = a + b*log(r).
        """
        # Prepare data for Good-Turing regression
        log_r = []
        log_Nr = []
        for r in sorted(self.freq_of_freqs):
            if r >= threshold:
                Nr = self.freq_of_freqs[r]
                if Nr > 0:
                    log_r.append(math.log(r))
                    log_Nr.append(math.log(Nr))

        # Fit regression if sufficient data points
        if len(log_Nr) >= 2:
            coeffs = np.polyfit(log_r, log_Nr, 1)
            b, a = coeffs  # log(Nr) = a + b*log(r)
        else:
            a = b = None

        processed_sent = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        ngrams = extract_ngrams(processed_sent, self.n)
        prob_dict = {}
        N = self.total_counts[self.n]
        N1 = self.freq_of_freqs.get(1, 0)

        for ngram in ngrams:
            if ngram in self.ngram_counts:
                r = self.ngram_counts[ngram]
                if r < threshold:  # Use raw counts for small r
                    Nr = self.freq_of_freqs[r]
                    Nr_plus_1 = self.freq_of_freqs.get(r + 1, 0)
                else:  # Use regression estimates for large r
                    if a is not None and b is not None:
                        # Calculate smoothed Nr = exp(a + b*log(r))
                        log_Nr = a + b * math.log(r)
                        Nr = math.exp(log_Nr)
                        log_Nr_plus_1 = a + b * math.log(r + 1)
                        Nr_plus_1 = math.exp(log_Nr_plus_1)
                    else:
                        Nr = self.freq_of_freqs.get(r, 0)
                        Nr_plus_1 = self.freq_of_freqs.get(r + 1, 0)
                
                # Avoid division by zero
                r_star = (r + 1) * Nr_plus_1 / Nr if Nr != 0 else 0.0
                prob = r_star / N if N != 0 else 0.0
            else:  # Unseen n-gram
                prob = N1 / N if N != 0 and N1 != 0 else 0.0
            
            prob_dict[ngram] = prob
        
        return prob_dict