import requests
from bs4 import BeautifulSoup, Tag
import os
import time
import random

BASE_URL = "https://www.law.cornell.edu"
UCC_INDEX = "https://www.law.cornell.edu/ucc"
SAVE_DIR = "ucc_articles"
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def safe_get(url, max_retries=5):
    for i in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                return resp
        except Exception as e:
            print(f"[WARN] Error fetching {url}: {e}. Retry {i+1}/{max_retries}")
            time.sleep(2 * (i+1))
    print(f"[ERROR] Failed to fetch {url} after {max_retries} retries.")
    return None

def get_article_links():
    resp = safe_get(UCC_INDEX)
    if resp is None:
        print(f"[ERROR] Could not fetch UCC_INDEX: {UCC_INDEX}")
        return []
    soup = BeautifulSoup(resp.content, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = a.get("href")
        if (
            isinstance(href, str)
            and href.startswith("/ucc/")
            and href.count("/") == 2
            and not href.endswith("ucc")
            and not href.endswith("/")
        ):
            links.add(BASE_URL + href)
    print(f"[DEBUG] Found article links: {links}")
    return list(links)

def get_section_links(article_url, article_code):
    resp = safe_get(article_url)
    if resp is None:
        print(f"[ERROR] Could not fetch article: {article_url}")
        return []
    soup = BeautifulSoup(resp.content, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = a.get("href")
        if (
            isinstance(href, str)
            and href.startswith(f"/ucc/{article_code}/")
            and href.count("/") == 3
            and (href.split("/")[-1].replace("-", "").isalnum())
        ):
            links.add(BASE_URL + href)
    print(f"[DEBUG] {article_code}: Found section links: {links}")
    return list(links)

def get_section_text(section_url):
    resp = safe_get(section_url)
    if not resp:
        return section_url, ""
    soup = BeautifulSoup(resp.content, "html.parser")
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else section_url
    content_div = soup.find("div", class_="section")
    content = ""
    if content_div and isinstance(content_div, Tag):
        # Lấy các <p> và <ol>/<ul> là con trực tiếp của div.section
        parts = []
        for child in content_div.children:
            if isinstance(child, Tag) and child.name == "p":
                text = child.get_text(separator="\n", strip=True)
                if text:
                    parts.append(text)
            elif isinstance(child, Tag) and child.name in ["ol", "ul"]:
                # Lấy từng mục trong danh sách
                for li in child.find_all("li", recursive=False):
                    li_text = li.get_text(separator="\n", strip=True)
                    if li_text:
                        parts.append(li_text)
        content = "\n\n".join(parts).strip()
        if not content:
            # Nếu không có <p> hoặc <ol>/<ul>, fallback lấy toàn bộ text như cũ
            content = content_div.get_text(separator="\n", strip=True)
    else:
        print(f"[DEBUG] No div.section at {section_url}, fallback to body")
        if soup.body:
            content = soup.body.get_text(separator="\n", strip=True)
            if not content.strip():
                print(f"[DEBUG] Empty body for {section_url}")
                print(soup.prettify()[:1000])
        else:
            content = ""
            print(f"[DEBUG] No <body> tag at {section_url}")
    return title, content

def crawl_all():
    article_links = get_article_links()
    print(f"Found {len(article_links)} articles.")
    for article_url in article_links:
        article_code = article_url.rstrip("/").split("/")[-1]
        article_dir = os.path.join(SAVE_DIR, article_code)
        os.makedirs(article_dir, exist_ok=True)
        section_links = get_section_links(article_url, article_code)
        print(f"Crawling {article_code}: {len(section_links)} sections")
        if not section_links:
            print(f"[WARN] No section links found for {article_code} at {article_url}")
        for section_url in section_links:
            section_id = section_url.rstrip("/").split("/")[-1]
            file_path = os.path.join(article_dir, f"{section_id}.txt")
            if os.path.exists(file_path):
                print(f"[SKIP] {article_code}/{section_id}.txt already exists.")
                continue
            title, content = get_section_text(section_url)
            if content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(title + "\n\n" + content)
                print(f"Saved {article_code}/{section_id}.txt")
            else:
                print(f"[WARN] No content for {article_code}/{section_id} at {section_url}")
            time.sleep(random.uniform(1.5, 3.5))  # Sleep random để tránh bị chặn
        print(f"Done {article_code}")

if __name__ == "__main__":
    crawl_all()