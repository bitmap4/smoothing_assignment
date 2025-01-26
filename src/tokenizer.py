# import re
# from typing import List

# class Tokenizer:
#     # Starting quotes.
#     STARTING_QUOTES = [
#         (re.compile("([«“‘„]|[`]+)", re.U), r" \1 "),
#         (re.compile(r"^\""), r"``"),
#         (re.compile(r"(``)"), r" \1 "),
#         (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
#         (re.compile(r"(?i)(\')(?!re|ve|ll|m|t|s|d|n)(\w)\b", re.U), r"\1 \2"),
#     ]

#     # Ending quotes.
#     ENDING_QUOTES = [
#         (re.compile("([»”’])", re.U), r" \1 "),
#         (re.compile(r"''"), " '' "),
#         (re.compile(r'"'), " '' "),
#         (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
#         (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
#     ]

#     URL_RULE = [
#         (re.compile(r'(?:https?|ftp)://[^\s]+'), r' \g<0> '),
#     ]

#     # Punctuation.
#     PUNCTUATION = [
#         (re.compile(r'([^\.])(\.)([\]\)}>"\'' 
#                     "»”’ " 
#                     r"]*)\s*$", re.U), r"\1 \2 \3 "),
#         (re.compile(r"([:,])([^\d ])"), r" \1 \2"),
#         (re.compile(r"([:,])$"), r" \1 "),
#         (
#             re.compile(r"\.{2,}", re.U),
#             r" \g<0> ",
#         ),  # See https://github.com/nltk/nltk/pull/2322
#         (re.compile(r"[;@#$%&]"), r" \g<0> "),
#         (
#             re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
#             r"\1 \2\3 ",
#         ),  # Handles the final period.
#         (re.compile(r"[?!]"), r" \g<0> "),
#         (re.compile(r"([^'])' "), r"\1 ' "),
#         (
#             re.compile(r"[*]", re.U),
#             r" \g<0> ",
#         ),  # See https://github.com/nltk/nltk/pull/2322
#     ]

#     # Pads parentheses
#     PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")

#     DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

#     CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(?#X)(not)\b")]

#     def tokenize(self, text: str) -> List[str]:
#         for regexp, substitution in self.STARTING_QUOTES:
#             text = regexp.sub(substitution, text)

#         for regexp, substitution in self.URL_RULE:
#             text = regexp.sub(substitution, text)

#         for regexp, substitution in self.PUNCTUATION:
#             text = regexp.sub(substitution, text)

#         # Handles parentheses.
#         regexp, substitution = self.PARENS_BRACKETS
#         text = regexp.sub(substitution, text)

#         # Handles double dash.
#         regexp, substitution = self.DOUBLE_DASHES
#         text = regexp.sub(substitution, text)

#         # add extra space to make things easier
#         text = " " + text + " "

#         for regexp, substitution in self.ENDING_QUOTES:
#             text = regexp.sub(substitution, text)

#         for regexp in self.CONTRACTIONS2:
#             text = regexp.sub(r" \1 \2 ", text)

#         return text.split()

#! ----------------------------------------------------------- !#

# import stanza
# from typing import List
# import os

# # Download the model only if not already downloaded
# if not os.path.exists(os.path.expanduser('~/stanza_resources/en')):
#     stanza.download('en')

# nlp = stanza.Pipeline(
#     'en',
#     processors='tokenize',
#     logging_level='ERROR',
#     download_method=stanza.DownloadMethod.REUSE_RESOURCES
# )

# def tokenize(text: str) -> List[List[str]]:
#     sentences = nlp(text).sentences
#     # [word.text for sent in sentences for word in sent.words]
#     return [[word.text for word in sent.words] for sent in sentences]

#! ----------------------------------------------------------- !#

# import nltk
# from typing import List

# nltk.download('punkt_tab', quiet=True)

# def tokenize(text: str) -> List[List[str]]:
#     return [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]

#! ----------------------------------------------------------- !#

import re
def tokenize(text):

    text = re.sub(r'http\S+', '<URL>', text)
    text = re.sub(r'www\S+', '<URL>', text)
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-za-z0-9.-]+\.[a-z]{2,}', '<MAILID>', text)

    text = re.sub(r'[^\@\#\.\w\?\!\s:-]', '', text)
    text = re.sub(f'-', ' ', text)
    text = re.sub(r'_', ' ', text)

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n*', '', text)
    text = re.sub(r'\.+', '.', text)

    abbreviations = re.findall(r'\b([A-Z]([a-z]){,2}\.)', text)
    if(abbreviations):
        abbreviations_set = set((list(zip(*abbreviations))[0]))

        for word in abbreviations_set:
            pattern = r'\b' + re.escape(word)
            text = re.sub(pattern, word.strip('.'), text)

    text = re.sub(r'#\w+\b', '<HASHTAG>', text)

    text = re.sub(r'@\w+\b', '<MENTION>', text)

    text = re.sub(r'\b\d+\b', '<NUM>', text)

    sentences = re.split(r'[.!?:]+', text)
    
    sentences = [sentence.strip() for sentence in sentences]

    dummy = []
    for sentence in sentences:
        current_word = ''
        tokens_in_sentence = []
        for char in sentence:
            if char != ' ':
                current_word += char
            elif current_word:
                tokens_in_sentence.append(current_word)
                current_word = ''
        if current_word:
            tokens_in_sentence.append(current_word)
        dummy.append(tokens_in_sentence)

    sentences = dummy

    sentences = [[word for word in sentence if word != ''] for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence]

    return sentences

#! ----------------------------------------------------------- !#

# import nltk
# import re

# def preprocess(text, Gutenberg=False):
#     lines= text.split("\n")
#     cleaned_lines= []

#     if Gutenberg:
#         skip= True
#         for line in lines:
#             if "***START OF THE PROJECT GUTENBERG EBOOK" in line.upper():
#                 skip= False
#                 continue
#             if "***END OF THE PROJECT GUTENBERG EBOOK" in line.upper():
#                 break
            
#             line= re.sub(r"CHAPTER\s+\d+", "", line, flags=re.IGNORECASE)
#             line= re.sub(r"--+", " ", line)
#             line= re.sub(r"_+", "", line)

#             if not skip:
#                 cleaned_lines.append(line.strip())

#     else:
#         for line in lines:
#             line= re.sub(r"--+", " ", line)
#             cleaned_lines.append(line.strip())

#     text= " ".join(cleaned_lines)
#     sentences= nltk.sent_tokenize(text)
#     cleaned_sentences= []

#     for sent in sentences:
#         sent= re.sub(r"\*+", "", sent)
#         cleaned_sentences.append(sent.strip())
        
#     return " ".join(cleaned_sentences)

# def tokenize(text, lower=False):
#     tokens= nltk.sent_tokenize(text)
#     for i in range(len(tokens)):
#         if lower:
#             tokens[i]= tokens[i].lower()
#         tokens[i]= nltk.word_tokenize(tokens[i])
#     return tokens



if __name__ == '__main__':
    text = input("your text: ")
    # text = preprocess(text)
    sentences = tokenize(text)
    print(f"Tokenized sentences: {sentences}")