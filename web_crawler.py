#!/usr/bin/env python3
"""
Web Crawler for AI Document Assistant - USCIS Laws and Policy
Extracts maximum text content from USCIS.gov and saves to a comprehensive text file for RAG processing
"""

import requests
from bs4 import BeautifulSoup
import time
import os
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path
import argparse
from typing import List, Dict, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class USCISWebCrawler:
    def __init__(self, output_dir: str = "crawled_data", delay: float = 1.5, max_pages: int = 50):
        """
        Initialize the USCIS web crawler for maximum content extraction
        
        Args:
            output_dir: Directory to save crawled text files
            delay: Delay between requests in seconds
            max_pages: Maximum number of pages to crawl (increased for more content)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.visited_urls: Set[str] = set()
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text for USCIS content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common USCIS navigation elements but keep more content
        text = re.sub(r'(Skip to main content|U\.S\. flag|Official Government Website|Secure Website|Español|Multilingual Resources|Sign In|Create Account|Menu|Close menu|Breadcrumb|Return to top|Facebook|X, formerly known as Twitter|YouTube|Instagram|LinkedIn|Email|Contact USCIS|U\.S\. Department of Homeland Security Seal|Agency description|Important links|Looking for U\.S\. government information and services\?|Visit USA\.gov|Was this page helpful\?|Yes|No|This page was not helpful because the content|Select a reason|has too little information|has too much information|is confusing|is out of date|other|How can the content be improved\?|To protect your privacy, please do not include any personal information in your feedback|Review our Privacy Policy)', '', text, flags=re.IGNORECASE)
        
        # Remove very short lines but keep more content
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 15]
        
        # Remove lines that are just punctuation or numbers
        lines = [line for line in lines if not re.match(r'^[\d\s\-\.]+$', line)]
        
        return '\n'.join(lines)
    
    def extract_text_from_page(self, url: str) -> Dict[str, str]:
        """
        Extract maximum text content from a single USCIS webpage
        
        Returns:
            Dictionary with 'title' and 'content' keys
        """
        try:
            logger.info(f"📄 Crawling: {url}")
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove only essential non-content elements
            for element in soup(["script", "style"]):
                element.decompose()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Extract maximum content - try multiple strategies
            content_parts = []
            
            # Strategy 1: Main content areas
            main_selectors = [
                'main',
                'article',
                '[role="main"]',
                '.content',
                '.main-content',
                '#content',
                '#main',
                '.post-content',
                '.entry-content',
                '.usa-main-content',
                '.usa-content',
                '.usa-grid',
                '.usa-width-one-whole',
                '.usa-width-two-thirds',
                '.usa-width-one-half',
                '.usa-section',
                '.usa-layout-docs__main',
                '.usa-layout-docs__content'
            ]
            
            for selector in main_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if text and len(text) > 50:
                        content_parts.append(text)
            
            # Strategy 2: All paragraphs and headings
            if not content_parts:
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 20:
                        content_parts.append(text)
            
            # Strategy 3: Body content as fallback
            if not content_parts:
                body = soup.find('body')
                if body:
                    content_parts.append(body.get_text())
            
            # Combine all content
            full_content = '\n\n'.join(content_parts)
            cleaned_content = self.clean_text(full_content)
            
            return {
                'title': title,
                'content': cleaned_content,
                'url': url
            }
            
        except Exception as e:
            logger.error(f"❌ Error crawling {url}: {e}")
            return {'title': '', 'content': '', 'url': url}
    
    def get_links_from_page(self, url: str, base_domain: str) -> List[str]:
        """Extract links from a page that belong to the same domain"""
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                
                # Only include links from the same domain
                if urlparse(full_url).netloc == base_domain:
                    # Filter out only the most obvious non-content URLs
                    if not any(skip in full_url.lower() for skip in [
                        'javascript:', 'mailto:', 'tel:', '#', '/search', '/contact', '/feedback'
                    ]):
                        links.append(full_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"❌ Error extracting links from {url}: {e}")
            return []
    
    def crawl_website(self, start_url: str, site_name: str) -> List[Dict[str, str]]:
        """
        Crawl the USCIS website starting from the given URL for maximum content
        
        Args:
            start_url: Starting URL for crawling
            site_name: Name for the site (used in filename)
            
        Returns:
            List of dictionaries containing page data
        """
        logger.info(f"🕷️ Starting comprehensive crawl of {site_name} from {start_url}")
        
        base_domain = urlparse(start_url).netloc
        pages_to_crawl = [start_url]
        crawled_pages = []
        
        while pages_to_crawl and len(crawled_pages) < self.max_pages:
            current_url = pages_to_crawl.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            self.visited_urls.add(current_url)
            
            # Extract content from current page
            page_data = self.extract_text_from_page(current_url)
            if page_data['content'] and len(page_data['content']) > 50:  # Lower threshold to get more content
                crawled_pages.append(page_data)
                logger.info(f"✅ Crawled page {len(crawled_pages)}/{self.max_pages}: {page_data['title'][:50]}... (Content length: {len(page_data['content'])} chars)")
            
            # Get links for next pages
            if len(crawled_pages) < self.max_pages:
                new_links = self.get_links_from_page(current_url, base_domain)
                for link in new_links:
                    if link not in self.visited_urls and link not in pages_to_crawl:
                        pages_to_crawl.append(link)
            
            # Respect rate limiting
            time.sleep(self.delay)
        
        logger.info(f"✅ Finished crawling {site_name}: {len(crawled_pages)} pages")
        return crawled_pages
    
    def save_to_file(self, pages: List[Dict[str, str]], site_name: str) -> str:
        """Save all crawled pages to a comprehensive text file"""
        if not pages:
            logger.warning(f"⚠️ No content to save for {site_name}")
            return ""
        
        # Create filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{site_name}_comprehensive_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        total_content_length = sum(len(page['content']) for page in pages)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# USCIS Laws and Policy - Comprehensive Crawled Data\n")
            f.write(f"# Source: https://www.uscis.gov/laws-and-policy\n")
            f.write(f"# Crawled on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total pages: {len(pages)}\n")
            f.write(f"# Total content length: {total_content_length:,} characters\n")
            f.write(f"# Average content per page: {total_content_length // len(pages):,} characters\n\n")
            
            for i, page in enumerate(pages, 1):
                f.write(f"## Page {i}: {page['title']}\n")
                f.write(f"URL: {page['url']}\n")
                f.write(f"Content Length: {len(page['content']):,} characters\n")
                f.write("-" * 80 + "\n")
                f.write(page['content'])
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        logger.info(f"💾 Saved {len(pages)} pages ({total_content_length:,} total characters) to {filepath}")
        return str(filepath)

def main():
    parser = argparse.ArgumentParser(description="USCIS Web Crawler for Maximum Content Extraction")
    parser.add_argument("--output", "-o", default="crawled_data", help="Output directory")
    parser.add_argument("--delay", "-d", type=float, default=1.5, help="Delay between requests (seconds)")
    parser.add_argument("--max-pages", "-m", type=int, default=50, help="Maximum pages to crawl (increased for more content)")
    
    args = parser.parse_args()
    
    # Initialize crawler
    crawler = USCISWebCrawler(
        output_dir=args.output,
        delay=args.delay,
        max_pages=args.max_pages
    )
    
    # USCIS Laws and Policy URL
    uscis_url = "https://www.uscis.gov/laws-and-policy"
    site_name = "uscis_laws_policy"
    
    try:
        logger.info(f"🚀 Starting comprehensive crawl of USCIS Laws and Policy")
        logger.info(f"📊 Target: Up to {args.max_pages} pages with maximum content extraction")
        pages = crawler.crawl_website(uscis_url, site_name)
        if pages:
            filepath = crawler.save_to_file(pages, site_name)
            total_chars = sum(len(page['content']) for page in pages)
            logger.info(f"✅ Successfully crawled USCIS -> {filepath}")
            logger.info(f"📈 Total content extracted: {total_chars:,} characters from {len(pages)} pages")
        else:
            logger.warning(f"⚠️ No content found for USCIS")
    except Exception as e:
        logger.error(f"❌ Error crawling USCIS: {e}")
    
    logger.info("🎉 USCIS comprehensive crawling completed!")

if __name__ == "__main__":
    main() 