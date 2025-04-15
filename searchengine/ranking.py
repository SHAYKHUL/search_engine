import math
from typing import List, Tuple, Dict
from indexing import Indexing
from nltk.corpus import wordnet
from fuzzywuzzy import fuzz
import re
import spacy
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter
import matplotlib.pyplot as plt

class AdvancedRanking:
    def __init__(self, indexer: Indexing, k1: float = 1.5, b: float = 0.75, damping_factor: float = 0.85,
                 content_weight: float = 0.7, pagerank_weight: float = 0.3):
        self.indexer = indexer
        self.k1 = k1
        self.b = b
        self.avg_doc_length = self.indexer.calculate_avg_doc_length()
        self.inverted_index, self.doc_lengths = self.indexer.get_inverted_index_and_doc_lengths()
        self.nlp = None  # SpaCy model will be loaded when needed
        self.damping_factor = damping_factor
        self.content_weight = content_weight
        self.pagerank_weight = pagerank_weight
        self.page_rank_scores = self.calculate_page_rank_scores()

    def get_spacy_model(self):
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        return self.nlp

    @lru_cache(maxsize=None)
    def expand_query(self, query: str) -> List[str]:
        expanded_query = set()
        exact_phrases = set(re.findall(r'"(.*?)"', query))

        for term in self.indexer.tokenize(query):
            expanded_query.add(term)
            synonyms = set(self.get_synonyms(term))
            expanded_query |= synonyms
            related_terms = set(self.get_related_terms(term))
            expanded_query |= related_terms

        expanded_query |= exact_phrases

        return list(expanded_query)

    def calculate_idf(self, term: str) -> float:
        num_docs_with_term = len(self.inverted_index.get(term, {}))
        num_docs = len(self.doc_lengths)
        return math.log((num_docs - num_docs_with_term + 0.5) / (num_docs_with_term + 0.5) + 1.0)

    def calculate_bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        idfs = np.vectorize(self.calculate_idf)(query_terms)
        tfs = np.array([self.calculate_tf(term, doc_id) for term in query_terms])
        doc_lengths = np.array([self.doc_lengths[doc_id] for _ in query_terms])

        numerators = tfs * (self.k1 + 1)
        denominators = tfs + self.k1 * (1 - self.b + self.b * (doc_lengths / (self.avg_doc_length + 1e-6)))

        non_zero_denominators = denominators != 0
        scores = np.sum(idfs * (numerators / denominators * non_zero_denominators))

        return scores

    def calculate_tf(self, term: str, doc_id: int) -> float:
        doc_info = self.inverted_index.get(term, {}).get(doc_id, {})
        return doc_info.get('frequency', 0)

    def rank_documents(self, query: str, fields: List[str] = None, boost_terms: Dict[str, float] = None) -> List[Tuple[int, float]]:
        query_terms = self.expand_query(query)
        scores = []

        with ThreadPoolExecutor() as executor:
            doc_ids = list(self.doc_lengths.keys())
            futures = {executor.submit(self.calculate_bm25_score, query_terms, doc_id): doc_id for doc_id in doc_ids}

            for future in as_completed(futures):
                doc_id = futures[future]
                bm25_score = future.result()
                scores.append((doc_id, bm25_score))

        sorted_results = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_results

    def combine_scores(self, content_score: float, pagerank_score: float) -> float:
        combined_score = (self.content_weight * content_score) + (self.pagerank_weight * pagerank_score)
        return combined_score

    def calculate_page_rank_scores(self) -> Dict[int, float]:
        graph = nx.DiGraph()

        graph.add_nodes_from(self.doc_lengths.keys())

        for doc_id, links in self.inverted_index.items():
            for link_id in links:
                graph.add_edge(doc_id, link_id)

        page_rank_scores = nx.pagerank(graph, alpha=self.damping_factor)

        return page_rank_scores

    def get_synonyms(self, term: str) -> List[str]:
        synonyms = set()
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    def get_related_terms(self, term: str) -> List[str]:
        # Replace with actual implementation based on your context
        return []

    def tokenize_with_phrases(self, text: str) -> List[str]:
        tokens = set(re.findall(r'\b\w+\b', text))
        return list(tokens)

    def apply_boosting(self, query_terms: List[str], boost_terms: Dict[str, float]) -> List[str]:
        boosted_query = []
        for term in query_terms:
            if term in boost_terms:
                boosted_query.extend([term] * int(boost_terms[term]))
            else:
                boosted_query.append(term)
        return boosted_query

    def fuzzy_match(self, term: str, query_term: str) -> bool:
        return fuzz.ratio(term.lower(), query_term.lower()) > 80

    def exact_match_boosting(self, query_terms: List[str], exact_match_terms: List[str]) -> List[str]:
        boosted_query = []
        for term in query_terms:
            if term in exact_match_terms:
                boosted_query.extend([term] * 3)
            else:
                boosted_query.append(term)
        return boosted_query

    def improved_phrase_matching(self, query_terms: List[str], doc_text: str) -> float:
        doc_tokens = self.tokenize_with_phrases(doc_text)
        phrase_counter = Counter(doc_tokens)
        phrase_matches = sum(phrase_counter[term] for term in query_terms)
        return phrase_matches

    def query_rewriting(self, query: str) -> str:
        rewritten_query = query.replace("important_term", "important_term OR synonym1 OR synonym2")
        return rewritten_query

    def concept_based_search(self, query_terms: List[str], doc_id: int) -> float:
        score = 0
        for term in query_terms:
            related_terms = self.get_related_terms(term)
            for related_term in related_terms:
                if related_term in self.inverted_index and doc_id in self.inverted_index[related_term]:
                    score += 1
        return score

if __name__ == "__main__":
    indexer = Indexing()
    advanced_ranker = AdvancedRanking(indexer)
    
    graph = nx.DiGraph()
    graph.add_nodes_from(advanced_ranker.page_rank_scores.keys())
    for doc_id, links in advanced_ranker.inverted_index.items():
        for link_id in links:
            graph.add_edge(doc_id, link_id)

    pos = nx.spring_layout(graph)  # You can choose a different layout if needed
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black",
            arrowsize=10, font_weight="bold", edge_color="gray", linewidths=0.5)

    # Add PageRank values as labels
    labels = {doc_id: f"PR: {score:.2f}" for doc_id, score in advanced_ranker.page_rank_scores.items()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_color="red")

    plt.title("PageRank Graph")
    
    # Save the graph as an image file
    plt.savefig("pagerank_graph.png")
    plt.close()  # Close the plot to avoid displaying it interactively


    boost_terms = {"important_term": 2, "another_term": 3}
    query = 'exact "search query" here'
    fields = ["title", "content", "url"]
    exact_match_terms = ["exact", "search", "query"]

    query_terms = advanced_ranker.expand_query(query)
    query_terms = advanced_ranker.apply_boosting(query_terms, boost_terms)
    query_terms = advanced_ranker.exact_match_boosting(query_terms, exact_match_terms)

    results = advanced_ranker.rank_documents(query, fields=fields, boost_terms=boost_terms)

    for rank, (doc_id, score) in enumerate(results, start=1):
        document_tuple = indexer.get_document_by_id(doc_id)
        print(f"Rank {rank}: Document ID {doc_id}, Score: {score:.4f}")
        print(f"URL: {document_tuple[1]}, Title: {document_tuple[2]}")
        print(f"Content: {document_tuple[3]}")
        print("----")
