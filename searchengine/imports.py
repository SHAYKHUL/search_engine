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
