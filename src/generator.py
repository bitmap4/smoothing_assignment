import argparse
from tokenizer import tokenize
from models import Base, LaplaceModel, GoodTuringModel, InterpolationModel
import heapq

parser = argparse.ArgumentParser(description='Generate next word based on input')
parser.add_argument('--method', '-m', type=str, choices=['i', 'g', 'l'], help='smoothing method (l=laplace, g=good-turing, i=interpolation)')
parser.add_argument('-n', type=int, help='n-gram order', default=3)
parser.add_argument('-k', type=int, help='number of candidates for next word', default=3)
parser.add_argument('corpus_path', type=str, help='path to the corpus')
args = parser.parse_args()

print("Loading corpus...")
with open(args.corpus_path) as file:
    text = file.read().lower().strip()

sentences = tokenize(text)

LM =    Base(args.n) if not args.method else \
        LaplaceModel(args.n) if args.method == 'l' else \
        GoodTuringModel(args.n) if args.method == 'g' else \
        InterpolationModel(args.n)
print("Training model...")
LM.train(sentences)
sent = tokenize(input("input sequence: ").lower())[-1]
sent = ['<s>']*(args.n-1) + sent

context = sent[-args.n+1:] if args.n != 1 else []

heap = []
for word in LM.vocab:
    prob_dict = LM.get_probabilities(context + [word])
    prob = prob_dict[tuple(context + [word])]
    heapq.heappush(heap, (prob, word))
    if len(heap) > args.k:
        heapq.heappop(heap)

print("Top k candidates:")
for prob, word in sorted(heap, reverse=True):
    print(f"{word}: {prob}\n")