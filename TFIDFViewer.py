"""
Use:
Receives the paths of two files, obtains its ids from the index, computes the tf-idf vectors for the corresponding
documents, optionally prints the vectors and finally computes their cosine similarity
Receives two paths of files to compare (the paths have to be the ones used when indexing the files)

Input: full path of two documents and its index name. Documents cannot be moved after
creating the index

Example:
python .\TFIDFViewer.py --index test --files D:\Documents\Data_Science\IRRS\lab2\docs\1 D:\Documents\Data_Science\IRRS\lab2\docs\1
"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import argparse

import numpy as np

__author__ = "bejar"


def search_file_by_path(client, index, path):
    """
    Returns the id of a document in the index (the path has to
    be the exact full path where the documents were when indexed,
    not just a filename)

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q("match", path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f"File [{path}] not found")
    else:
        return lfiles[0].meta.id


def document_term_vector(client, index, id):
    """
    returns two lists of pairs, the first one is (term, term
    frequency in the document), the second one is (term, term frequency in the index). Both lists
    are alphabetically ordered by term.

    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(
        index=index, id=id, fields=["text"], positions=False, term_statistics=True
    )

    file_td = {}
    file_df = {}

    if "text" in termvector["term_vectors"]:
        for t in termvector["term_vectors"]["text"]["terms"]:
            file_td[t] = termvector["term_vectors"]["text"]["terms"][t]["term_freq"]
            file_df[t] = termvector["term_vectors"]["text"]["terms"][t]["doc_freq"]
    return sorted(file_td.items()), sorted(file_df.items())


def toTFIDF(client, index, file_id):
    """
    Returns a list of pairs (term, weight) representing the
    document with the given docid. It:
    1. First gets two lists with term document frequency and term index frequency
    2. Gets the number of documents in the index.
    3. Then finally creates every pair (term, TFIDF) entry of the vector to be returned.

    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = []
    for (t, w), (_, df) in zip(file_tv, file_df):
        #
        # Calculate tf-idf and fill the value
        #
        pass

    return normalize(tfidfw)


def print_term_weigth_vector(twv):
    """
    COMPLEATE

    prints one line for each entry in the given vector
    of the form (term, weight).

    Prints the term vector and the correspondig weights
    :param twv:
    :return:
    """
    #
    # Program something here
    #
    pass


def normalize(tw):
    """
    COMPLEATE

    compute the norm of the vector (square root of the
    sums of components squared) and divide the whole vector by it, so that the resulting vector
    has norm (length) 1. Complete this function.

    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """
    #
    # Program something here
    #
    return None


def cosine_similarity(tw1, tw2):
    """
    COMPLEATE

    Computes the cosine similarity between two weight vectors, terms are alphabetically ordered
    :param tw1:
    :param tw2:
    :return:
    """
    #
    # Program something here
    #
    return 0


def doc_count(client, index):
    """
    Returns the number of documents in an index

    :param client:
    :param index:
    :return:
    """
    return int(client.cat.count(index=[index], format="json")[0]["count"])


if __name__ == "__main__":
    # read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=None, required=True, help="Index to search")
    parser.add_argument(
        "--files",
        default=None,
        required=True,
        nargs=2,
        help="Paths of the files to compare",
    )
    parser.add_argument(
        "--print", default=False, action="store_true", help="Print TFIDF vectors"
    )

    args = parser.parse_args()

    index = args.index

    file1 = args.files[0]
    file2 = args.files[1]

    client = Elasticsearch(request_timeout=1000, hosts=["http://localhost:9200"])

    try:

        # Get the files ids
        file1_id = search_file_by_path(client, index, file1)
        file2_id = search_file_by_path(client, index, file2)

        print(file1_id, file2_id)

        # Compute the TF-IDF vectors
        file1_tw = toTFIDF(client, index, file1_id)
        file2_tw = toTFIDF(client, index, file2_id)

        if args.print:
            print(f"TFIDF FILE {file1}")
            print_term_weigth_vector(file1_tw)
            print("---------------------")
            print(f"TFIDF FILE {file2}")
            print_term_weigth_vector(file2_tw)
            print("---------------------")

        """
        print(f"Similarity = {cosine_similarity(file1_tw, file2_tw):3.5f}")
        """

    except NotFoundError:
        print(f"Index {index} does not exists")
