#! /usr/bin/env python3

import sys, re, os, pickle, argparse
from collections import Counter
from collections import defaultdict
from functools import reduce


# remove non alphabets at the beginning or end of a word
# NOTE: this assumes the text is English!
def clean(iterable):
    m1 = map(lambda x: re.sub("^[^a-zA-Z]+|[^a-zA-Z]+$", "", x), iterable)
    return map(lambda x: x.lower(), m1)


def main(dir_root, output_file, stop_file=None, max_word_length=20):
    stop_set = set()
    documents = list()
    word_ids = dict()
    word_list = list()

    # term frequency
    tf = Counter()
    # store a set of documents for each word occurred in
    idf = defaultdict(set)

    # load optional stop word list if exists
    if stop_file:
        with open(stop_file, 'r') as f:
            m = map(lambda x: re.split('\s+', x), f)
            stop_set = set(map(lambda x: x.lower(), reduce(lambda acc, se: acc.union(se), m, set())))

    # store each document's path
    for root, dirs, files in os.walk(dir_root):
        for x in files:
            documents.append(os.path.join(root, x))

    doc_id = 0
    total = len(documents)
    try:
        # process each document, store the cleaned, lower-cased vocab and assign each word an ID
        # store a term frequency dictionary for each document: tf[docid, id] = freq
        # store a document frequency for each word: idf[id] = {set of documents}
        for doc in documents:
            with open(doc, 'r', encoding="iso-8859-1") as d:
                print("processing %s, %d/%d" % (doc, doc_id+1, total))
                for line in d:
                    v = filter(lambda x: x not in stop_set and 0 < len(x) <= max_word_length, clean(re.split('\s+', line)))
                    for word in v:
                        # word_id is the number when it first appears.
                        if word in word_ids:
                            id = word_ids[word]
                        else:
                            # this is a new word, assign an id
                            id = len(word_ids)
                            word_ids[word] = id
                            word_list.append(word)

                        # store document term frequency
                        tf[doc_id, id] += 1
                        idf[id].add(doc_id)
            doc_id += 1
    except KeyboardInterrupt:
        print("Program interrupted ...")
    finally:
        # serialise model as a file
        print("saving model as:", output_file)
        with open(output_file, 'wb') as f:
            pickle.dump((documents, word_ids, word_list, tf, idf), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=sys.argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", help="Input file/folder for index building")
    parser.add_argument("output", help="output file for index object")
    parser.add_argument("-s", "--stop_list", type=str, default=None, help='stop word list')
    parser.add_argument("-m", "--max_word_length", type=int, default=20, help='ignore words longer than this')

    args = parser.parse_args()
    main(args.input, args.output, args.stop_list, args.max_word_length)
