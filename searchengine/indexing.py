# indexing.py
import sqlite3
import json
from collections import defaultdict
from math import log
import re
from operator import itemgetter
from typing import List

class Indexing:
    def __init__(self):
        self.inverted_index = defaultdict(lambda: defaultdict(dict))
        self.doc_lengths = defaultdict(float)

    def create_database(self):
        conn = sqlite3.connect('search_engine.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT,
                content TEXT,
                named_entities TEXT,
                quality_score REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY,
                inverted_index TEXT,
                doc_lengths TEXT
            )
        ''')
        c.execute('INSERT OR IGNORE INTO metadata (id) VALUES (1)')
        conn.commit()
        conn.close()

    def insert_data(self, url, title, content, named_entities=None, quality_score=0):
        conn = sqlite3.connect('search_engine.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO documents (url, title, content, named_entities, quality_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (url, title, content, named_entities, quality_score))
        conn.commit()
        conn.close()

    def build_index(self):
        conn = sqlite3.connect('search_engine.db')
        c = conn.cursor()
        c.execute('SELECT id, content FROM documents')
        rows = c.fetchall()

        for row in rows:
            doc_id, doc_content = row
            self.index_document(doc_id, doc_content)

        # Store the inverted index and doc_lengths in the metadata table
        c.execute('UPDATE metadata SET inverted_index=?, doc_lengths=? WHERE id=1',
                  (json.dumps(self.inverted_index), json.dumps(self.doc_lengths)))

        conn.commit()
        conn.close()

    def index_document(self, doc_id, doc_content):
        terms = self.tokenize(doc_content)
        term_freq = defaultdict(int)

        for term in terms:
            term_freq[term] += 1

        doc_length = sum(term_freq.values())
        self.doc_lengths[doc_id] = doc_length

        for term, frequency in term_freq.items():
            self.inverted_index[term][doc_id] = {'frequency': frequency}

    def tokenize(self, text):
        terms = re.findall(r'\b\w+\b', text.lower())

        stop_words = set(['the', 'and', 'is', 'in', 'it', 'of', 'to', 'for', 'with', 'on'])
        terms = [term for term in terms if term not in stop_words]

        terms = [self.stem_word(word) for word in terms]

        return terms

    def stem_word(self, word):
        if word.endswith('s'):
            return word[:-1]
        return word

    def calculate_avg_doc_length(self):
        total_doc_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)

        # Check if num_docs is zero to avoid division by zero
        return total_doc_length / num_docs if num_docs != 0 else 0

    def calculate_idf(self, term):
        num_docs_with_term = len(self.inverted_index[term])
        num_docs = len(self.doc_lengths)
        return log((num_docs - num_docs_with_term + 0.5) / (num_docs_with_term + 0.5) + 1.0)

    def search(self, query):
        query_terms = self.tokenize(query)
        scores = defaultdict(float)

        for term in query_terms:
            idf = self.calculate_idf(term)

            for doc_id, info in self.inverted_index[term].items():
                tf = info['frequency']
                scores[doc_id] += tf * idf

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def optimize_index(self):
        avg_doc_length = self.calculate_avg_doc_length()

        for term, postings in list(self.inverted_index.items()):
            if len(postings) < 2:
                del self.inverted_index[term]
                continue

            idf = self.calculate_idf(term)

            postings = {doc_id: info for doc_id, info in postings.items() if info['frequency'] > 1}

            compressed_postings = self.compress_postings(postings)

            self.inverted_index[term] = {'idf': idf, 'postings': compressed_postings}

    def compress_postings(self, postings):
        compressed_postings = []
        postings_list = sorted(postings.items(), key=itemgetter(0))

        if not postings_list:
            return compressed_postings  # Return an empty list if postings_list is empty

        for i in range(len(postings_list) - 1):
            doc_id, freq = postings_list[i]
            next_doc_id, next_freq = postings_list[i + 1]

            delta_doc_id = next_doc_id - doc_id
            compressed_postings.append((delta_doc_id, freq))

        last_doc_id, last_freq = postings_list[-1]
        compressed_postings.append((last_doc_id, last_freq))

        return compressed_postings

    def decompress_postings(self, compressed_postings):
        postings = []
        doc_id = 0

        for delta_doc_id, freq in compressed_postings:
            doc_id += delta_doc_id
            postings.append((doc_id, freq))

        return postings

    def get_document_by_id(self, doc_id):
        conn = sqlite3.connect('search_engine.db')
        c = conn.cursor()
        c.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = c.fetchone()
        conn.close()
        return row

    def get_documents_by_ids(self, doc_ids: List[int]):
        conn = sqlite3.connect('search_engine.db')
        c = conn.cursor()
        query = 'SELECT * FROM documents WHERE id IN ({})'.format(','.join(map(str, doc_ids)))
        c.execute(query)
        rows = c.fetchall()
        conn.close()
        return rows

    def get_inverted_index_and_doc_lengths(self):
        conn = sqlite3.connect('search_engine.db')
        c = conn.cursor()
        c.execute('SELECT inverted_index, doc_lengths FROM metadata WHERE id = 1')
        row = c.fetchone()
        conn.close()

        if row:
            inverted_index_str, doc_lengths_str = row
            inverted_index = json.loads(inverted_index_str) if inverted_index_str else {}
            doc_lengths = json.loads(doc_lengths_str) if doc_lengths_str else {}
            return inverted_index, doc_lengths
        else:
            return {}, {}
