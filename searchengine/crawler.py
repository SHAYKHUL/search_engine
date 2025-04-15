import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import HtmlResponse
from scrapy.utils.log import configure_logging
import logging
import os
import hashlib
import spacy
import json
from urllib.parse import urlparse, urljoin
from indexing import Indexing
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule
from bs4 import BeautifulSoup
from extruct.w3cmicrodata import MicrodataExtractor
from textblob import TextBlob
from readability import Document
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from textstat import flesch_reading_ease, coleman_liau_index
import networkx as nx
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
from urllib import robotparser
from scrapy.selector import Selector

nltk.download('punkt')
nltk.download('stopwords')

# Set file paths for visited URLs and hashes
VISITED_URLS_FILE = 'visited_urls.txt'
VISITED_HASHES_FILE = 'visited_hashes.txt'

# Load the spaCy models
nlp = spacy.load("en_core_web_sm")
nlp_advanced = spacy.load("en_core_web_sm")

# Configure Scrapy logging
configure_logging(install_root_handler=False)
logging.basicConfig(
    filename='scrapy.log',
    format='%(levelname)s: %(message)s',
    level=logging.DEBUG
)

class MyItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    named_entities = scrapy.Field()
    quality_score = scrapy.Field()
    pagerank_score = scrapy.Field()
    sentiment = scrapy.Field()
    snippet = scrapy.Field()
    named_entities_advanced = scrapy.Field()
    sentiment_polarity = scrapy.Field()
    keyphrases = scrapy.Field()

class MyCrawler(scrapy.Spider):
    name = 'mycrawler'

    custom_settings = {
        'CONCURRENT_REQUESTS': 64,
        'DOWNLOAD_DELAY': 0.05,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter',
        'DOWNLOAD_DELAY': 0.1,
        'CONCURRENT_REQUESTS': 32,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 0.5,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 8.0,
        'FEED_FORMAT': 'jsonlines',
        'FEED_URI': 'output.json',
        'HTTPCACHE_ENABLED': True,
        'HTTPCACHE_IGNORE_HTTP_CODES': [404, 503],
        'HTTPCACHE_EXPIRATION_SECS': 2592000,
        'DEPTH_LIMIT': 5,
        'COOKIES_ENABLED': False,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 522, 524, 408, 429],
        'RETRY_TIMES': 3,
        'ROBOTSTXT_OBEY': True,
        'FOCUSED_TOPICS': ['technology', 'science', 'business'],  # Define your focused topics
    }

    rules = (
        Rule(LinkExtractor(allow=('/articles/',)), callback='parse_article'),
        Rule(LinkExtractor(deny=('/login/', '/register/'))),
    )

    def __init__(self, *args, **kwargs):
        super(MyCrawler, self).__init__(*args, **kwargs)
        self.visited_urls = self.load_visited_set(VISITED_URLS_FILE)
        self.visited_hashes = self.load_visited_set(VISITED_HASHES_FILE)
        self.indexer = Indexing()
        self.graph = nx.DiGraph()
        self.domain_page_count = {}

    def start_requests(self):
        seed_url_file = 'seed_urls.txt'

        if not os.path.exists(seed_url_file):
            self.logger.error(f"Seed URL file '{seed_url_file}' not found.")
            return

        with open(seed_url_file, 'r') as file:
            seed_urls = file.read().splitlines()

        for url in seed_urls:
            yield scrapy.Request(url, callback=self.parse, dont_filter=True)

    def parse(self, response):
        if not isinstance(response, HtmlResponse):
            return

        domain = urlparse(response.url).netloc

        if domain in self.domain_page_count and self.domain_page_count[domain] >= 50:
            self.logger.info(f'Reached maximum limit for domain {domain}. Skipping further pages.')
            return

        self.domain_page_count[domain] = self.domain_page_count.get(domain, 0) + 1

        content_hash = hashlib.md5(response.body).hexdigest()

        if content_hash in self.visited_hashes:
            self.logger.info(f'Skipping duplicate content with hash: {content_hash}')
            return

        self.visited_hashes.add(content_hash)

        if response.url in self.visited_urls:
            self.logger.info(f'Skipping already visited URL: {response.url}')
            return

        self.visited_urls.add(response.url)

        item = MyItem()
        item['url'] = response.url
        item['title'] = response.css('title::text').get()
        item['content'] = response.css('p::text').getall()

        doc = nlp(" ".join(item['content']))
        named_entities = [ent.text for ent in doc.ents]
        item['named_entities'] = named_entities

        quality_score = self.assess_content_quality(response.url, item['title'], item['content'])
        item['quality_score'] = quality_score

        pagerank_score = self.calculate_pagerank_score(response.url)
        item['pagerank_score'] = pagerank_score

        sentiment_analysis = TextBlob(" ".join(item['content']))
        item['sentiment'] = sentiment_analysis.sentiment.polarity

        item['snippet'] = self.extract_snippet(item['content'])

        # Perform advanced NLP analysis
        advanced_nlp_results = self.analyze_advanced_nlp(" ".join(item['content']))
        item['named_entities_advanced'] = advanced_nlp_results['named_entities_advanced']
        item['sentiment_polarity'] = advanced_nlp_results['sentiment_polarity']
        item['keyphrases'] = advanced_nlp_results['keyphrases']

        yield item

        self.indexer.create_database()
        self.indexer.insert_data(
            item['url'],
            item['title'],
            '\n'.join(item['content']),
            named_entities=json.dumps(item['named_entities']),
            quality_score=item['quality_score']
        )

        self.graph.add_node(response.url)
        for link in self.extract_links(response, doc):
            self.graph.add_edge(response.url, link)

        links_to_follow = self.extract_links(response, doc)

        for link in links_to_follow:
            parsed_url = urlparse(link)
            if parsed_url.scheme in ['http', 'https']:
                yield response.follow(link, callback=self.parse)

        if len(self.visited_urls) % 10 == 0:
            self.save_visited_set(self.visited_urls, VISITED_URLS_FILE)
            self.save_visited_set(self.visited_hashes, VISITED_HASHES_FILE)

    def parse_article(self, response):
        pass

    def extract_links(self, response, doc):
        links = set(response.css('a::attr(href)').getall())

        # Filter out invalid links (e.g., javascript:, mailto:)
        links = {link for link in links if not link.startswith(('javascript:', 'mailto:'))}

        microdata_extractor = MicrodataExtractor()
        microdata = microdata_extractor.extract(response.text)
        microdata_links = set(item['properties']['url'] for item in microdata if 'url' in item.get('properties', {}))

        all_links = links.union(microdata_links)

        prioritized_links = self.prioritize_links(response.url, all_links, doc)

        return prioritized_links

    def prioritize_links(self, base_url, links, doc):
        prioritized_links = []

        for link in links:
            abs_link = urljoin(base_url, link)
            link_doc = nlp(abs_link)

            similarity = doc.similarity(link_doc)

            common_entities = set(doc.ents).intersection(link_doc.ents)
            entity_overlap_score = len(common_entities)

            link_score = similarity * entity_overlap_score

            prioritized_links.append({'url': abs_link, 'score': link_score})

        prioritized_links.sort(key=lambda x: x['score'], reverse=True)

        return [link['url'] for link in prioritized_links]

    def assess_content_quality(self, url, title, content):
        try:
            full_text = f"{title} {' '.join(content)}"
            flesch_score = flesch_reading_ease(full_text)
            coleman_liau_score = coleman_liau_index(full_text)
            language = detect(full_text)
            tokens = word_tokenize(full_text)
            tokens = [word.lower() for word in tokens if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            freq_dist = FreqDist(tokens)
            unique_word_ratio = len(freq_dist) / len(tokens)
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', full_text)
            avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if len(sentences) > 0 else 0

            quality_score = (
                (flesch_score / 100) * 0.3 +
                (coleman_liau_score / 20) * 0.3 +
                (unique_word_ratio * 0.2) +
                (1 if language == 'en' else 0.2) +
                (avg_sentence_length / 30) * 0.2
            )

            return quality_score
        except Exception as e:
            self.logger.error(f"Error in content quality analysis for {url}: {e}")
            return 0

    def calculate_pagerank_score(self, url):
        try:
            page_rank_scores = nx.pagerank(self.graph, alpha=0.85)
        except nx.NetworkXError:
            page_rank_scores = {}

        return page_rank_scores.get(url, 0.0)

    def extract_snippet(self, content):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', " ".join(content))
        snippet = " ".join(sentences[:2])
        return snippet

    def analyze_advanced_nlp(self, text):
        doc_advanced = nlp_advanced(text)

        named_entities_advanced = [ent.text for ent in doc_advanced.ents]

        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        sentiment_polarity = sentiment_scores['compound']

        r = Rake()
        r.extract_keywords_from_text(text)
        keyphrases = r.get_ranked_phrases()

        return {
            'named_entities_advanced': named_entities_advanced,
            'sentiment_polarity': sentiment_polarity,
            'keyphrases': keyphrases,
        }

    def find_sitemaps(self, response):
        # Initialize a robot parser
        rp = robotparser.RobotFileParser()
        rp.set_url(urljoin(response.url, '/robots.txt'))
        rp.read()

        # Check if the user-agent is allowed to fetch the robots.txt file
        if rp.can_fetch('*', '/robots.txt'):
            # Parse robots.txt and get sitemap URLs
            sitemaps = self.extract_sitemaps_from_robots(response.url + '/robots.txt')
        else:
            self.logger.warning(f"Cannot fetch robots.txt from {response.url}")
            sitemaps = []

        # Extract sitemap URLs from meta tags
        sitemaps += self.extract_sitemaps_from_meta(response)

        # Add a common location for sitemap.xml
        sitemaps.append(urljoin(response.url, '/sitemap.xml'))

        # Explore other ways to find sitemaps, e.g., by looking for <loc> tags in the HTML content
        sitemaps += self.extract_sitemaps_from_html(response)

        return sitemaps

    def extract_sitemaps_from_robots(self, robots_url):
        # Parse robots.txt and get sitemap URLs
        try:
            robots_content = self.download_page(robots_url)
            rp = robotparser.RobotFileParser()
            rp.parse(robots_content.decode('utf-8').splitlines())
            return rp.site_maps()
        except Exception as e:
            self.logger.error(f"Error extracting sitemaps from robots.txt: {e}")
            return []

    def extract_sitemaps_from_meta(self, response):
        sitemaps = []
        try:
            # Parse HTML content using Scrapy selector
            selector = Selector(response)
            
            # Extract sitemap URLs from meta tags
            sitemap_elements = selector.xpath('//link[@rel="sitemap"]/@href').getall()
            sitemaps += [urljoin(response.url, sitemap) for sitemap in sitemap_elements]
        except Exception as e:
            self.logger.error(f"Error extracting sitemaps from meta tags: {e}")
        return sitemaps

    def extract_sitemaps_from_html(self, response):
        sitemaps = []
        try:
            # Parse HTML content using Scrapy selector
            selector = Selector(response)
            
            # Extract sitemap URLs from <loc> tags in the HTML content
            loc_elements = selector.xpath('//loc/text()').getall()
            sitemaps += [urljoin(response.url, loc) for loc in loc_elements]
        except Exception as e:
            self.logger.error(f"Error extracting sitemaps from HTML: {e}")
        return sitemaps

    def download_page(self, url):
        # Function to download a page content
        try:
            with self.crawler.engine.open_sitemap(url) as response:
                return response.body
        except Exception as e:
            self.logger.error(f"Error downloading page {url}: {e}")
            return b''

    def load_visited_set(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            return set()

    def save_visited_set(self, visited_set, file_path):
        with open(file_path, 'w') as file:
            file.write('\n'.join(visited_set))

    def save_state(self):
        self.save_visited_set(self.visited_urls, VISITED_URLS_FILE)
        self.save_visited_set(self.visited_hashes, VISITED_HASHES_FILE)
        self.save_sitemap_state()

    def save_sitemap_state(self):
        with open('sitemap_state.json', 'w') as file:
            sitemap_state = {
                'visited_sitemaps': list(self.visited_sitemaps),
                'last_sitemap_update': self.last_sitemap_update,
            }
            json.dump(sitemap_state, file)

    def load_sitemap_state(self):
        try:
            with open('sitemap_state.json', 'r') as file:
                sitemap_state = json.load(file)
                self.visited_sitemaps = set(sitemap_state.get('visited_sitemaps', []))
                self.last_sitemap_update = sitemap_state.get('last_sitemap_update', 0)
        except FileNotFoundError:
            self.visited_sitemaps = set()
            self.last_sitemap_update = 0

def run_crawler():
    process = CrawlerProcess()
    try:
        process.crawl(MyCrawler)
        process.start()
    except KeyboardInterrupt:
        print("Crawling interrupted. Saving state and optimizing index...")
        MyCrawler().save_state()
        MyCrawler().indexer.build_index()
        MyCrawler().indexer.optimize_index()
    finally:
        process.stop()

if __name__ == '__main__':
    run_crawler()
