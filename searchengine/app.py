from flask import Flask, render_template, request, jsonify, abort
import redis
from ranking import AdvancedRanking
from indexing import Indexing
import requests
from datetime import datetime
import nltk
import string
import re
import html
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import time
from functools import lru_cache
import logging
import traceback
import asyncio
import aiohttp
import hashlib
import json

# Download NLTK data
nltk.download('punkt')

# Flask app
app = Flask(__name__)

# Redis setup
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Logger setup
logging.basicConfig(level=logging.INFO)

# Load indexer & ranker
indexer = Indexing()
indexer.build_index()
advanced_ranker = AdvancedRanking(indexer)

# SentenceTransformer for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Wikipedia API config
WIKIPEDIA_API_URL = 'https://en.wikipedia.org/w/api.php'
USER_AGENT = 'AlgolizenSearch/1.0'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        start_time = datetime.now()

        query = request.form['query'].strip()
        page_number = int(request.args.get('page', 1))

        if not query:
            return render_template('index.html')

        # --- Track co-occurring queries ---
        words = query.lower().split()
        for word in set(words):
            redis_client.sadd(f"related:{word}", query)

        # --- Get topic info from cache or Wikipedia ---
        topic_info = get_from_cache(f"topic_info:{query}")
        if not topic_info:
            topic_info = asyncio.run(async_get_topic_information(query))
            set_to_cache(f"topic_info:{query}", topic_info)

        # --- Get search results from cache or compute them ---
        ranking_cache_key = f"ranking_results:{query}:{page_number}"
        results = get_from_cache(ranking_cache_key)
        if not results:
            results = advanced_ranker.rank_documents(query)
            results = [(doc_id, score) for doc_id, score in results if score > 5.0]
            set_to_cache(ranking_cache_key, results)

        results_per_page = 10
        start_index = (page_number - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = results[start_index:end_index]

        if not paginated_results:
            return render_template('no_results.html', query=query)

        ranked_documents = []
        all_content = []
        for rank, (doc_id, score) in enumerate(paginated_results, start=start_index + 1):
            document = indexer.get_document_by_id(doc_id)
            if document:
                snippet = generate_snippet(document[3], query)
                ranked_documents.append({
                    'rank': rank,
                    'url': document[1],
                    'title': document[2],
                    'content': snippet,
                    'score': f'{score:.4f}'
                })
                all_content.append(document[3])

        answer = generate_direct_answer(query, all_content)

        end_time = datetime.now()
        time_taken = end_time - start_time

        return render_template(
            'results.html',
            query=query,
            ranked_documents=ranked_documents,
            topic_info=topic_info,
            direct_answer=answer,
            page_number=page_number,
            num_results=len(results),
            time_taken=time_taken
        )

    except Exception as e:
        logging.error(f"Error in search: {str(e)}")
        logging.error(traceback.format_exc())
        return render_template('error.html', error_message="Something went wrong while processing your request.")

@app.route('/related')
def related_queries():
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Missing query'}), 400

    # Collect related queries from Redis
    related_set = set()
    for word in set(query.lower().split()):
        related_set.update(redis_client.smembers(f"related:{word}"))

    related_set.discard(query)
    return jsonify(list(related_set))



@app.route('/load_more')
async def load_more():
    try:
        query = request.args.get('query')
        page_number = int(request.args.get('page', 1))

        # Asynchronous document ranking request
        results = await async_rank_documents(query)
        results = [(doc_id, score) for doc_id, score in results if score > 0.0]

        results_per_page = 10
        start_index = (page_number - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_results = results[start_index:end_index]

        if not paginated_results:
            return jsonify({'next_page': page_number + 1, 'results': []})

        response_data = {
            'next_page': page_number + 1,
            'results': [
                {
                    'rank': rank,
                    'url': indexer.get_document_by_id(doc_id)[1],
                    'title': indexer.get_document_by_id(doc_id)[2],
                    'content': generate_snippet(indexer.get_document_by_id(doc_id)[3], query),
                    'score': f'{score:.4f}'
                }
                for rank, (doc_id, score) in enumerate(paginated_results, start=start_index + 1)
            ]
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in load_more: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'Failed to load more results.'})


# --- Redis Caching Functions ---
def get_from_cache(cache_key):
    cached_data = redis_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    return None


def set_to_cache(cache_key, data, ttl=3600):
    redis_client.setex(cache_key, ttl, json.dumps(data))


# --- Wikipedia Extract ---
async def async_get_topic_information(query):
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'titles': query,
            'prop': 'extracts|info|pageprops',
            'exintro': True,
            'explaintext': True,
            'inprop': 'url'
        }
        headers = {'User-Agent': USER_AGENT}

        async with aiohttp.ClientSession() as session:
            async with session.get(WIKIPEDIA_API_URL, params=params, headers=headers) as response:
                data = await response.json()
                page = next(iter(data['query']['pages'].values()), None)

                if page and 'extract' in page:
                    return {
                        'title': page['title'],
                        'summary': page['extract'],
                        'fullurl': page.get('fullurl', ''),
                        'pageprops': page.get('pageprops', {})
                    }
        return None

    except Exception as e:
        logging.error(f"Error in async Wikipedia API call: {str(e)}")
        return None


# --- Snippet Generator ---
def generate_snippet(content, query, max_sentences=3, max_length=300):
    if not content or not query:
        return ""

    query_terms = [term.strip().lower() for term in re.split(r'[,\s]+', query) if term.strip()]
    sentences = sent_tokenize(content)

    query_embedding = model.encode([query])
    sentence_embeddings = model.encode(sentences)
    similarities = cosine_similarity(query_embedding, sentence_embeddings).flatten()

    scored_sentences = []
    for i, sentence in enumerate(sentences):
        similarity_score = similarities[i]
        if similarity_score > 0:
            match_count = sum(1 for term in query_terms if term in sentence.lower())
            final_score = similarity_score + (match_count / len(query_terms))
            scored_sentences.append((final_score, sentence.strip()))

    top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:max_sentences]
    snippet = " ".join([highlight_terms(sent, query_terms) for _, sent in top_sentences])

    if len(snippet) > max_length:
        snippet = snippet[:max_length] + "..."

    return snippet


# --- Direct Answer Generator ---
def generate_direct_answer(query, documents, max_sentences=2):
    all_sentences = []
    for doc in documents:
        all_sentences.extend(sent_tokenize(doc))

    if not all_sentences:
        return ""

    query_embedding = model.encode([query])
    sentence_embeddings = model.encode(all_sentences)
    similarities = cosine_similarity(query_embedding, sentence_embeddings).flatten()

    top_indices = similarities.argsort()[-max_sentences:][::-1]
    top_sentences = [all_sentences[i] for i in top_indices]

    return " ".join(top_sentences)


# --- Highlighter ---
def highlight_terms(text, query_terms, max_highlights=5):
    if not text or not query_terms:
        return html.escape(text)

    text = html.escape(text)  # Prevent HTML injection
    query_terms = sorted(set(query_terms), key=len, reverse=True)

    # Track which spans have already been marked
    marked = []
    highlights_added = 0

    for term in query_terms:
        if highlights_added >= max_highlights:
            break

        pattern = re.compile(r'\b(' + re.escape(term) + r')\b', flags=re.IGNORECASE)
        matches = list(pattern.finditer(text))

        for match in matches:
            start, end = match.span()

            # Skip if overlapping with already marked spans
            if any(start < m_end and end > m_start for m_start, m_end in marked):
                continue

            # Insert <mark> tags
            text = (
                text[:start]
                + f"<mark>{text[start:end]}</mark>"
                + text[end:]
            )

            # Update all marked spans
            offset = len("<mark></mark>")
            marked.append((start, end + offset))
            highlights_added += 1

            break  # Only one highlight per term to avoid duplication

    return text


# --- Asynchronous Document Ranking ---
async def async_rank_documents(query):
    return advanced_ranker.rank_documents(query)


# --- Error Handling ---
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_message="Page not found."), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_message="Internal server error. Please try again later."), 500


# --- Run Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
