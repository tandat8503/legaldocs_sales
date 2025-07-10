import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://www.law.cornell.edu"
ARTICLE2_URL = "https://www.law.cornell.edu/ucc/2"
SAVE_DIR = "chatbot/legal_data/ucc_article2_sections"
os.makedirs(SAVE_DIR, exist_ok=True)

def crawl_article2_sections():
    resp = requests.get(ARTICLE2_URL)
    soup = BeautifulSoup(resp.content, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if isinstance(href, str) and href.startswith("/ucc/2/2-"):
            links.add(href)
    for href in links:
        section_url = BASE_URL + href
        section_resp = requests.get(section_url)
        section_soup = BeautifulSoup(section_resp.content, "html.parser")
        title_tag = section_soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else href

        content_div = section_soup.find("div", class_="section")
        content = content_div.get_text(separator="\n", strip=True) if content_div else ""
        filename = href.split("/")[-1] + ".txt"
        with open(os.path.join(SAVE_DIR, filename), "w", encoding="utf-8") as f:
            f.write(title + "\n\n" + content)
        print(f"Saved {filename}")

if __name__ == "__main__":
    crawl_article2_sections()