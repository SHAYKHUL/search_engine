#app.py
from flask import Flask, render_template, request, jsonify
from ranking import AdvancedRanking
from indexing import Indexing
import requests
from datetime import datetime

app = Flask(__name__)

# Initialize the Indexing and AdvancedRanking instances
indexer = Indexing()
indexer.build_index()  # Ensure that the index is built before using AdvancedRanking
advanced_ranker = AdvancedRanking(indexer)  # Update initialization

# Initialize Wikipedia API with a custom user agent
WIKIPEDIA_API_URL = 'https://en.wikipedia.org/w/api.php'
USER_AGENT = 'YourApp/1.0'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    start_time = datetime.now()  # Record the start time of the search

    query = request.form['query'].strip()
    page_number = int(request.args.get('page', 1))

    if not query:
        return render_template('index.html')

    # Get information from Wikipedia API for the search topic
    topic_info = get_topic_information(query)

    # Use AdvancedRanking for ranking documents
    results = advanced_ranker.rank_documents(query)

    # Filter out results with a score of 0.0000
    results = [(doc_id, score) for doc_id, score in results if score > 5.0000]

    # Paginate the results
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
            ranked_documents.append({
                'rank': rank,
                'url': document[1],
                'title': document[2],
                'content': document[3],
                'score': f'{score:.4f}'
            })

    end_time = datetime.now()  # Record the end time of the search
    time_taken = end_time - start_time  # Calculate the time taken

    return render_template(
        'results.html',
        query=query,
        ranked_documents=ranked_documents,
        topic_info=topic_info,
        page_number=page_number,
        num_results=len(results),  # Add the number of results
        time_taken=time_taken  # Add the time taken
    )

@app.route('/load_more')
def load_more():
    query = request.args.get('query')
    page_number = int(request.args.get('page', 1))
    
    # Use the same logic as in the /search route to get paginated results
    results = advanced_ranker.rank_documents(query)
    results = [(doc_id, score) for doc_id, score in results if score > 0.0000]
    
    results_per_page = 10  # Adjust the number of results per page as needed
    start_index = (page_number - 1) * results_per_page
    end_index = start_index + results_per_page
    paginated_results = results[start_index:end_index]

    if not paginated_results:
        return jsonify({'next_page': page_number + 1, 'results': []})

    # Build the JSON response
    response_data = {
        'next_page': page_number + 1,
        'results': [
            {
                'rank': rank,
                'url': indexer.get_document_by_id(doc_id)[1],
                'title': indexer.get_document_by_id(doc_id)[2],
                'content': indexer.get_document_by_id(doc_id)[3],
                'score': f'{score:.4f}'
            }
            for rank, (doc_id, score) in enumerate(paginated_results, start=start_index + 1)
        ]
    }

    return jsonify(response_data)

def get_topic_information(query):
    # Use requests library to send a GET request to Wikipedia API
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

    # Extract information from the API response
    page = next(iter(data['query']['pages'].values()), None)

    if page and 'extract' in page:
        # Check if the page is a disambiguation page
        if 'disambiguation' in page.get('categories', ''):
            # If disambiguation, try to find the most relevant option
            options = page.get('links', [])
            if options:
                first_option = options[0]
                return {
                    'title': first_option['title'],
                    'summary': f"This might refer to {first_option['title']}. "
                               f"Please specify your search or choose a more specific term.",
                    'fullurl': f"https://en.wikipedia.org/wiki/{first_option['title']}"
                }

        # Retrieve additional page properties
        page_props = page.get('pageprops', {})

        return {
            'title': page['title'],
            'summary': page['extract'],
            'fullurl': page.get('fullurl', ''),
            'pageprops': page_props
        }
    else:
        return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
