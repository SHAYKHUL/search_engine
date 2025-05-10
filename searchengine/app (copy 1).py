from flask import Flask, render_template, request, jsonify
from ranking import AdvancedRanking
from indexing import Indexing
import requests
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
import string
import re

# Download necessary NLTK data
nltk.download('punkt')

app = Flask(__name__)

# Initialize the Indexing and AdvancedRanking instances
indexer = Indexing()
indexer.build_index()
advanced_ranker = AdvancedRanking(indexer)

# Wikipedia API config
WIKIPEDIA_API_URL = 'https://en.wikipedia.org/w/api.php'
USER_AGENT = 'AlgolizenSearch/1.0'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    start_time = datetime.now()

    query = request.form['query'].strip()
    page_number = int(request.args.get('page', 1))

    if not query:
        return render_template('index.html')

    topic_info = get_topic_information(query)

    results = advanced_ranker.rank_documents(query)
    results = [(doc_id, score) for doc_id, score in results if score > 5.0000]

    results_per_page = 10
    start_index = (page_number - 1) * results_per_page
    end_index = start_index + results_per_page
    paginated_results = results[start_index:end_index]

    if not paginated_results:
        return render_template('no_results.html', query=query)

    ranked_documents = []
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

    end_time = datetime.now()
    time_taken = end_time - start_time

    return render_template(
        'results.html',
        query=query,
        ranked_documents=ranked_documents,
        topic_info=topic_info,
        page_number=page_number,
        num_results=len(results),
        time_taken=time_taken
    )

@app.route('/load_more')
def load_more():
    query = request.args.get('query')
    page_number = int(request.args.get('page', 1))

    results = advanced_ranker.rank_documents(query)
    results = [(doc_id, score) for doc_id, score in results if score > 0.0000]

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

def get_topic_information(query):
    params = {
        'action': 'query',
        'format': 'json',
        'titles': query,
        'prop': 'extracts|info|pageprops',
        'exintro': True,
        'explaintext': True,
        'inprop': 'url'
    }

    headers = {
        'User-Agent': USER_AGENT
    }

    response = requests.get(WIKIPEDIA_API_URL, params=params, headers=headers)
    data = response.json()

    page = next(iter(data['query']['pages'].values()), None)

    if page and 'extract' in page:
        page_props = page.get('pageprops', {})
        return {
            'title': page['title'],
            'summary': page['extract'],
            'fullurl': page.get('fullurl', ''),
            'pageprops': page_props
        }
    else:
        return None

# --- SMART SNIPPET GENERATOR ---
def generate_snippet(content, query, max_sentences=3):
    if not content or not query:
        return ""

    query_terms = set(word.lower().strip(string.punctuation) for word in query.split())
    sentences = sent_tokenize(content)
    scored_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        words = set(word.strip(string.punctuation) for word in sentence_lower.split())
        match_count = sum(1 for term in query_terms if term in words)
        if match_count > 0:
            score = match_count + len(set(words) & query_terms) / len(query_terms)
            scored_sentences.append((score, sentence.strip()))

    # Pick top N scored sentences
    top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:max_sentences]

    if not top_sentences:
        return content[:300] + "..."

    snippet = " ".join([highlight_terms(sent, query_terms) for _, sent in top_sentences])
    return snippet + "..."

def highlight_terms(text, terms):
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(rf'\b({re.escape(term)})\b', flags=re.IGNORECASE)
        text = pattern.sub(r'<mark>\1</mark>', text)
    return text

# --- MAIN ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
