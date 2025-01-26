import argparse
from tokenizer import tokenize
from random import shuffle
from models import LaplaceModel, GoodTuringModel, InterpolationModel, sentence_probability

# MODELS = {
#     ('l', 'pride.txt'): 'LM1',
#     ('g', 'pride.txt'): 'LM2',
#     ('i', 'pride.txt'): 'LM3',
#     ('l', 'alice.txt'): 'LM4',
#     ('g', 'alice.txt'): 'LM5',
#     ('i', 'alice.txt'): 'LM6',
# }

parser = argparse.ArgumentParser(description="Train and test n-gram language models")
parser.add_argument("--method", "-m", type=str, choices=['l', 'g', 'i'], help="smoothing method (l=laplace, g=good-turing, i=interpolation)", default='l')
parser.add_argument("-n", type=int, help="n-gram order", default=3)
parser.add_argument("--analyze", "-a", type=str, choices=['train', 'test'], help="analyze perplexity on either train or test data")
parser.add_argument("corpus_path", type=str, help="path to corpus file")
args = parser.parse_args()

with open(args.corpus_path, 'r') as file:
    text = file.read().lower().strip()
    sentences = tokenize(text)

LM =    LaplaceModel(args.n) if args.method == 'l' else \
        GoodTuringModel(args.n) if args.method == 'g' else \
        InterpolationModel(args.n)

if args.analyze:
    shuffle(sentences)
    train = sentences[:-1000]
    test = sentences[-1000:]

    LM.train(train)

    perplexities = []
    corpus_name = args.corpus_path.split('/')[-1]
    outputs = []
    for sent in (test if args.analyze == 'test' else train):
        prob_dict = LM.get_probabilities(sent)
        perplexities.append(LM.calculate_perplexity(prob_dict))
        outputs.append(f"{' '.join(sent)}\t{perplexities[-1]}")
    # with open(f'2023114009_{MODELS[(args.method, corpus_name)]}_{args.n}_{args.analyze}-avg-perplexity.txt', 'w') as f:
    #     f.write(f"{sum(perplexities)/len(perplexities)}")
    #     f.write('\n'.join(outputs))
    print(f"{sum(perplexities)/len(perplexities)}")
    print('\n'.join(outputs))

else:
    LM.train(sentences)

    test = tokenize(input("Enter test sentence: ").strip().lower())[0]
    prob_dict = LM.get_probabilities(test)
    perplexity = LM.calculate_perplexity(prob_dict)
    print(f"Probability: {sentence_probability(prob_dict)}")
    print(f"Perplexity: {perplexity}")