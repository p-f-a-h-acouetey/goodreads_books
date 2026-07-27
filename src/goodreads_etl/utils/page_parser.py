"""Field-level HTML/JSON-LD parsers for a single Goodreads book page.

Every function is pure (soup/text in, field out) so they're trivial to
unit-test with saved HTML fixtures.

Verified against live Goodreads markup: book pages embed a JSON-LD
<script type="application/ld+json"> block (@type "Book") plus a Next.js
__NEXT_DATA__ hydration payload; JSON-LD is the primary, most reliable
source, with page text and __NEXT_DATA__ as fallbacks for fields Goodreads
doesn't always put in the structured block (e.g. publisher, reading stats).
"""

from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

from src.goodreads_etl.utils.text_parser import clean_object, get_first_match, get_int


# ---------------------------------------------------------------------------
# Structured-data extraction (JSON-LD / Next.js hydration payload)
# ---------------------------------------------------------------------------

DEFAULT_JSON_LD_SCRIPT_TYPE = "application/ld+json"
DEFAULT_JSON_TYPE_KEY = "@type"
DEFAULT_JSON_GRAPH_KEY = "@graph"
DEFAULT_JSON_LD_VALID_TYPES = frozenset({"Book", "Product"})


def _iter_json_ld_blocks(*, soup: BeautifulSoup, script_type: str) -> list[dict | list]:
    """Parse and return every JSON-LD script block's decoded JSON payload.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        script_type: The `type` attribute string identifying JSON-LD tags.

    Returns:
        A list of decoded JSON objects (dicts or lists) extracted from the tags.
    """
    blocks: list[dict | list] = []
    for script in soup.find_all("script", type=script_type):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if data:
            blocks.append(data)
    return blocks


def _find_valid_book_entry(
    *,
    data: dict | list,
    type_key: str,
    graph_key: str,
    valid_types: frozenset[str],
) -> dict | None:
    """Search one decoded JSON-LD block for a valid Book or Product entry.

    Handles flat objects, lists of objects, or top-level `@graph` wrappers.

    Args:
        data: Decoded JSON data structure (dictionary or list).
        type_key: Key name used to declare entity type in JSON-LD (e.g. '@type').
        graph_key: Key name used for graph nodes array (e.g. '@graph').
        valid_types: Acceptable type strings indicating a valid book entry.

    Returns:
        The matched dictionary representing the book entry, or None if not found.
    """
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get(type_key) in valid_types:
                return item
        return None

    if isinstance(data, dict):
        if data.get(type_key) in valid_types:
            return data
        graph = data.get(graph_key)
        if isinstance(graph, list):
            for item in graph:
                if isinstance(item, dict) and item.get(type_key) in valid_types:
                    return item

    return None


def extract_json_ld(
    *,
    soup: BeautifulSoup,
    script_type: str = DEFAULT_JSON_LD_SCRIPT_TYPE,
    type_key: str = DEFAULT_JSON_TYPE_KEY,
    graph_key: str = DEFAULT_JSON_GRAPH_KEY,
    valid_types: frozenset[str] = DEFAULT_JSON_LD_VALID_TYPES,
) -> dict | None:
    """Extract the book's JSON-LD structured data block, if present.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        script_type: The `type` attribute value identifying JSON-LD `<script>` tags.
            Defaults to `'application/ld+json'`.
        type_key: Property key identifying schema entity types in JSON-LD.
            Defaults to `'@type'`.
        graph_key: Property key referencing graph nodes array in JSON-LD root.
            Defaults to `'@graph'`.
        valid_types: Set of acceptable `@type` values representing books.
            Defaults to `{'Book', 'Product'}`.

    Returns:
        The matched JSON-LD dictionary if a valid book node is found, otherwise None.
    """
    for block in _iter_json_ld_blocks(soup=soup, script_type=script_type):
        entry = _find_valid_book_entry(
            data=block, type_key=type_key, graph_key=graph_key, valid_types=valid_types
        )
        if entry:
            return entry
    return None


DEFAULT_NEXT_DATA_SCRIPT_ID = "__NEXT_DATA__"


def extract_next_data(
    *,
    soup: BeautifulSoup,
    script_id: str = DEFAULT_NEXT_DATA_SCRIPT_ID,
) -> dict | None:
    """Extract the Next.js __NEXT_DATA__ hydration payload, if present.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        script_id: The `id` attribute identifying the Next.js hydration tag.
            Defaults to `'__NEXT_DATA__'`.

    Returns:
        Decoded dictionary representing the Next.js state tree, or None if missing or invalid.
    """
    script = soup.find("script", id=script_id)
    if not script:
        return None

    raw = script.string or script.get_text(strip=True)
    if not raw:
        return None

    try:
        result: dict[Any, Any] | None = json.loads(raw)
        return result
    except json.JSONDecodeError:
        return None


def find_first_key(*, data: object, keys: frozenset[str]) -> str | None:
    """Recursively search a nested dict/list for the first non-empty string.

    Used to hunt for publisher-like fields inside the unpredictable
    `__NEXT_DATA__` tree.

    Args:
        data: The nested dictionary, list, or scalar object to traverse.
        keys: Target keys to match during tree traversal.

    Returns:
        The first matching non-empty string scalar value found, or None.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys and isinstance(value, str) and value.strip():
                return value
        for value in data.values():
            result = find_first_key(data=value, keys=keys)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_first_key(data=item, keys=keys)
            if result:
                return result

    return None


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

DEFAULT_DESCRIPTION_SELECTORS = (
    ("span", {"class": "Formatted"}),
    ("div", {"data-testid": "description"}),
)
DEFAULT_META_DESCRIPTION_ITEMPROP = "description"


def extract_description(
    *,
    soup: BeautifulSoup,
    description_selectors: tuple[tuple[str, dict], ...] = DEFAULT_DESCRIPTION_SELECTORS,
    meta_itemprop: str = DEFAULT_META_DESCRIPTION_ITEMPROP,
) -> str | None:
    """Extract the book description from known selectors or meta tags.

    Falls back to the page's `<meta itemprop="description">` tag if DOM selectors fail.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        description_selectors: Ordered sequence of `(tag, attribute_dict)` targets.
            Defaults to common Goodreads layout class names and test IDs.
        meta_itemprop: Value of `itemprop` on `<meta>` tag to check as a fallback.
            Defaults to `'description'`.

    Returns:
        Cleaned description text if extracted, otherwise None.
    """
    for tag, attrs in description_selectors:
        node = soup.find(tag, attrs=attrs)
        if node:
            return clean_object(value=node.get_text(separator="\n", strip=True))

    meta = soup.find("meta", itemprop=meta_itemprop)
    if meta:
        return clean_object(value=meta.get("content"))

    return None


# ---------------------------------------------------------------------------
# Genres
# ---------------------------------------------------------------------------

DEFAULT_GENRES_PATH_FRAGMENT = "/genres/"


def extract_genres(
    *,
    soup: BeautifulSoup,
    genres_path_fragment: str = DEFAULT_GENRES_PATH_FRAGMENT,
) -> str | None:
    """Extract genre names from genre links, de-duplicated, order preserved.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        genres_path_fragment: URL substring identifying genre anchor tags.
            Defaults to `'/genres/'`.

    Returns:
        Comma-separated string of unique genres in display order, or None if none found.
    """
    genres: list[str] = []

    for link in soup.find_all("a", href=True):
        href = str(link.get("href") or "")
        text = clean_object(value=link.get_text(strip=True))
        if genres_path_fragment in href and text is not None:
            genres.append(text)

    genres = list(dict.fromkeys(genres))
    return ", ".join(genres) if genres else None


# ---------------------------------------------------------------------------
# Author (book-page link, JSON-LD fallback)
# ---------------------------------------------------------------------------

DEFAULT_AUTHOR_PATH_FRAGMENT = "/author/"
DEFAULT_ROOT_URL = "https://www.goodreads.com"
DEFAULT_AUTHOR_KEY = "author"
DEFAULT_NAME_KEY = "name"


def _find_author_link(*, soup: BeautifulSoup, author_path_fragment: str, root_url: str) -> tuple[str, str] | None:
    """Return author name and absolute URL from the first matching anchor tag.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        author_path_fragment: URL substring matching author links (e.g. '/author/').
        root_url: Base domain string to construct full URLs for relative links.

    Returns:
        A tuple of `(author_name, absolute_author_url)`, or None if no link matches.
    """
    for link in soup.find_all("a", href=True):
        href = str(link.get("href") or "")
        if author_path_fragment not in href:
            continue

        name = clean_object(value=link.get_text(strip=True))
        if not name:
            continue

        absolute_url = href if href.startswith("http") else f"{root_url}{href}"
        return name, absolute_url

    return None


def _find_author_in_json_ld(*, json_ld: dict | None, author_key: str, name_key: str) -> str | None:
    """Return the primary author's name from JSON-LD without a URL fallback.

    Args:
        json_ld: Extracted JSON-LD structure dictionary.
        author_key: Key inside JSON-LD payload holding author info.
        name_key: Field key inside author payload storing display name.

    Returns:
        Cleaned author name string, or None if missing or unparseable.
    """
    if not json_ld:
        return None

    authors = json_ld.get(author_key, [])
    if isinstance(authors, dict):
        authors = [authors]

    for author in authors:
        if isinstance(author, dict):
            name = clean_object(value=author.get(name_key, ""))
            if name:
                return name

    return None


def extract_first_author_and_url(
    *,
    soup: BeautifulSoup,
    json_ld: dict | None,
    author_path_fragment: str = DEFAULT_AUTHOR_PATH_FRAGMENT,
    root_url: str = DEFAULT_ROOT_URL,
    author_key: str = DEFAULT_AUTHOR_KEY,
    name_key: str = DEFAULT_NAME_KEY,
) -> tuple[str | None, str | None]:
    """Extract the primary author's name and profile URL.

    The book page's anchor tag is preferred because it also gives us the
    author URL needed to enrich follower/book counts later; JSON-LD is
    only a name-only fallback when no author link is found on the page.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        json_ld: Previously extracted JSON-LD block (if available).
        author_path_fragment: URL substring identifying author profile links.
            Defaults to `'/author/'`.
        root_url: Base domain used to resolve relative author links.
            Defaults to `'https://www.goodreads.com'`.
        author_key: Key in JSON-LD object representing author payload.
            Defaults to `'author'`.
        name_key: Key in JSON-LD author dictionary holding display name.
            Defaults to `'name'`.

    Returns:
        A tuple of `(author_name, author_url)`. Values are None if not resolvable.
    """
    from_link = _find_author_link(
        soup=soup, author_path_fragment=author_path_fragment, root_url=root_url
    )
    if from_link:
        return from_link

    name = _find_author_in_json_ld(json_ld=json_ld, author_key=author_key, name_key=name_key)
    return (name, None) if name else (None, None)


# ---------------------------------------------------------------------------
# Page count
# ---------------------------------------------------------------------------

DEFAULT_NUM_PAGES_KEY = "numberOfPages"
DEFAULT_PAGES_PATTERNS = (
    re.compile(r"(\d[\d,]*)\s+pages?", re.IGNORECASE),
    re.compile(r"pages?\s*[:|-]?\s*(\d[\d,]*)", re.IGNORECASE),
)


def extract_num_pages(
    *,
    soup: BeautifulSoup,
    json_ld: dict | None,
    num_pages_key: str = DEFAULT_NUM_PAGES_KEY,
    pages_patterns: tuple[re.Pattern[str], ...] = DEFAULT_PAGES_PATTERNS,
) -> int | None:
    """Extract page count, preferring JSON-LD over scraping page text.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        json_ld: Previously extracted JSON-LD block (if available).
        num_pages_key: Key in JSON-LD object holding page count.
            Defaults to `'numberOfPages'`.
        pages_patterns: Regexes used to locate page numbers in raw text.

    Returns:
        Integer page count, or None if extraction fails.
    """
    if json_ld and json_ld.get(num_pages_key):
        return get_int(text=str(json_ld.get(num_pages_key)))

    page_text = soup.get_text(" ", strip=True)
    return get_int(text=get_first_match(text=page_text, patterns=pages_patterns))


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

DEFAULT_FORMAT_PATTERNS = (
    re.compile(r"\b(hardcover|paperback|kindle edition|ebook|audiobook)\b", re.IGNORECASE),
)


def extract_format(
    *,
    soup: BeautifulSoup,
    format_patterns: tuple[re.Pattern[str], ...] = DEFAULT_FORMAT_PATTERNS,
) -> str | None:
    """Extract the edition format (hardcover, paperback, ebook, etc.).

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        format_patterns: Regexes identifying known book format strings.

    Returns:
        Matched format name (e.g. 'Hardcover', 'Paperback'), or None.
    """
    page_text = clean_object(value=soup.get_text(" ", strip=True))
    return clean_object(value=get_first_match(text=page_text, patterns=format_patterns))


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

DEFAULT_SERIES_PATH_FRAGMENT = "/series/"
DEFAULT_SERIES_PATTERNS = (
    re.compile(r"\(([^()]+?)\s+#\d+(?:\.\d+)?\)", re.IGNORECASE),
    re.compile(r"Series\s*:\s*([^|]+)", re.IGNORECASE),
)


def extract_series(
    *,
    soup: BeautifulSoup,
    series_path_fragment: str = DEFAULT_SERIES_PATH_FRAGMENT,
    series_patterns: tuple[re.Pattern[str], ...] = DEFAULT_SERIES_PATTERNS,
) -> str | None:
    """Extract the series name a book belongs to, if any.

    Text is deliberately kept un-cleaned before regex matching, since
    clean_object() strips parenthetical content (e.g. "(#1)") which is the
    exact notation these patterns need to see.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        series_path_fragment: URL substring identifying series links.
            Defaults to `'/series/'`.
        series_patterns: Regexes capturing series title before volume numbers.

    Returns:
        Title of the series if found, otherwise None.
    """
    for link in soup.find_all("a", href=True):
        href = str(link.get("href") or "")
        raw_text = link.get_text(" ", strip=True)
        if series_path_fragment not in href or not raw_text:
            continue

        match = get_first_match(text=raw_text, patterns=series_patterns)
        if match:
            return clean_object(value=match)

    return None


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------

DEFAULT_AGGREGATE_RATING_KEY = "aggregateRating"
DEFAULT_NUM_REVIEWS_KEY = "reviewCount"
DEFAULT_REVIEWS_PATTERNS = (
    re.compile(r"(\d[\d,]*)\s+reviews?", re.IGNORECASE),
    re.compile(r"reviews?\s*[:|-]?\s*(\d[\d,]*)", re.IGNORECASE),
)


def extract_num_reviews(
    *,
    soup: BeautifulSoup,
    json_ld: dict | None,
    aggregate_rating_key: str = DEFAULT_AGGREGATE_RATING_KEY,
    num_reviews_key: str = DEFAULT_NUM_REVIEWS_KEY,
    reviews_patterns: tuple[re.Pattern[str], ...] = DEFAULT_REVIEWS_PATTERNS,
) -> int | None:
    """Extract total review count, preferring JSON-LD's aggregateRating.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        json_ld: Previously extracted JSON-LD block (if available).
        aggregate_rating_key: Key in JSON-LD holding rating summary.
            Defaults to `'aggregateRating'`.
        num_reviews_key: Field inside aggregate rating dictionary holding review count.
            Defaults to `'reviewCount'`.
        reviews_patterns: Regex fallback patterns if JSON-LD is missing.

    Returns:
        Integer count of reviews, or None if extraction fails.
    """
    if json_ld:
        aggregate = json_ld.get(aggregate_rating_key, {}) or {}
        if aggregate.get(num_reviews_key) is not None:
            return get_int(text=str(aggregate.get(num_reviews_key)))

    page_text = soup.get_text(" ", strip=True)
    return get_int(text=get_first_match(text=page_text, patterns=reviews_patterns))


# ---------------------------------------------------------------------------
# First published date
# ---------------------------------------------------------------------------

DEFAULT_DATE_PUBLISHED_KEY = "datePublished"
DEFAULT_PUBLISHED_PATTERNS = (
    re.compile(r"first published\s+(.+?)(?:\s*[|•]|\s*$)", re.IGNORECASE),
    re.compile(r"published\s+(.+?)(?:\s*[|•]|\s*$)", re.IGNORECASE),
)


def extract_first_published(
    *,
    soup: BeautifulSoup,
    json_ld: dict | None,
    published_patterns: tuple[re.Pattern[str], ...] = DEFAULT_PUBLISHED_PATTERNS,
    date_published_key: str = DEFAULT_DATE_PUBLISHED_KEY,
) -> str | None:
    """Extract first-publication date.

    Page text is checked first because Goodreads often shows the *original*
    first-published date there, while JSON-LD's datePublished sometimes
    reflects the specific edition's date instead.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        json_ld: Previously extracted JSON-LD block (if available).
        published_patterns: Regex patterns capturing original publication string.
        date_published_key: Fallback key in JSON-LD holding publication date.
            Defaults to `'datePublished'`.

    Returns:
        Publication date string, or None if not available.
    """
    page_text = soup.get_text(" ", strip=True)

    result = get_first_match(text=page_text, patterns=published_patterns)
    if result:
        return clean_object(value=result)

    if json_ld and json_ld.get(date_published_key):
        return clean_object(value=str(json_ld.get(date_published_key)))

    return None


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

DEFAULT_NEXT_DATA_PUBLISHER_KEYS = frozenset({"publisher", "publisherName", "imprint"})
DEFAULT_PUBLISHER_KEY = "publisher"
DEFAULT_PUBLISHER_PATTERNS = (
    re.compile(r"publisher\s*[:|-]?\s*([^|•\n]+)", re.IGNORECASE),
)


def _publisher_from_next_data(*, next_data: dict | None, keys: frozenset[str]) -> str | None:
    """Extract publisher string from Next.js payload.

    Args:
        next_data: Decoded Next.js hydration payload.
        keys: Target keys to search for inside Next.js object tree.

    Returns:
        Publisher string if located, otherwise None.
    """
    if not next_data:
        return None
    name = find_first_key(data=next_data, keys=keys)
    return clean_object(value=name) if name else None


def _publisher_from_json_ld(*, json_ld: dict | None, publisher_key: str, name_key: str) -> str | None:
    """Extract publisher string from JSON-LD schema block.

    Args:
        json_ld: Decoded JSON-LD payload dictionary.
        publisher_key: Top-level publisher field name inside JSON-LD payload.
        name_key: Sub-field key containing publisher's name.

    Returns:
        Publisher string if located, otherwise None.
    """
    if not json_ld:
        return None
    publisher = json_ld.get(publisher_key)
    if isinstance(publisher, dict):
        return clean_object(value=publisher.get(name_key, ""))
    if isinstance(publisher, str):
        return clean_object(value=publisher)
    return None


def _publisher_from_page_text(*, soup: BeautifulSoup, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    """Extract publisher string from raw page text using fallback regexes.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        patterns: Regex patterns matching publisher patterns.

    Returns:
        Publisher string if matched, otherwise None.
    """
    page_text = soup.get_text(" ", strip=True)
    result = get_first_match(text=page_text, patterns=patterns)
    return clean_object(value=result) if result else None


def extract_publisher(
    *,
    soup: BeautifulSoup,
    json_ld: dict | None,
    next_data: dict | None,
    next_data_publisher_keys: frozenset[str] = DEFAULT_NEXT_DATA_PUBLISHER_KEYS,
    publisher_key: str = DEFAULT_PUBLISHER_KEY,
    name_key: str = DEFAULT_NAME_KEY,
    publisher_patterns: tuple[re.Pattern[str], ...] = DEFAULT_PUBLISHER_PATTERNS,
) -> str | None:
    """Extract the publishing house: next_data, then JSON-LD, then page text.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        json_ld: Previously extracted JSON-LD block (if available).
        next_data: Previously extracted Next.js hydration payload (if available).
        next_data_publisher_keys: Keys to hunt for inside Next.js payload structure.
        publisher_key: Top-level publisher field name inside JSON-LD payload.
            Defaults to `'publisher'`.
        name_key: Child field name for publisher name string inside JSON-LD.
            Defaults to `'name'`.
        publisher_patterns: Regex fallback patterns evaluated against visible page text.

    Returns:
        Publisher name string if located, otherwise None.
    """
    return (
        _publisher_from_next_data(next_data=next_data, keys=next_data_publisher_keys)
        or _publisher_from_json_ld(json_ld=json_ld, publisher_key=publisher_key, name_key=name_key)
        or _publisher_from_page_text(soup=soup, patterns=publisher_patterns)
    )


# ---------------------------------------------------------------------------
# Reading stats (currently reading / want to read)
# ---------------------------------------------------------------------------

DEFAULT_CURRENTLY_READING_PATTERNS = (
    re.compile(r"(\d[\d,]*)\s+people?\s+(?:are\s+)?currently\_reading", re.IGNORECASE),
)
DEFAULT_WANT_TO_READ_PATTERNS = (
    re.compile(r"(\d[\d,]*)\s+people?\s+want\s+to\s+read", re.IGNORECASE),
)


def _search_sources_for_count(
    *,
    sources: tuple[str, ...],
    patterns: tuple[re.Pattern[str], ...],
) -> int:
    """Return the first non-zero count found across multiple text sources.

    Goodreads sometimes only exposes these counts in inline `<script>` JSON
    rather than the rendered text, so both the raw HTML and the rendered
    page text are checked.

    Args:
        sources: Sequence of text strings to scan.
        patterns: Regex patterns capturing reader metrics.

    Returns:
        Extracted integer value, or 0 if no count is found across sources.
    """
    for source in sources:
        count = get_int(text=get_first_match(text=source, patterns=patterns))
        if count:
            return count
    return 0


def extract_reading_stats(
    *,
    soup: BeautifulSoup,
    html_text: str,
    currently_reading_patterns: tuple[re.Pattern[str], ...] = DEFAULT_CURRENTLY_READING_PATTERNS,
    want_to_read_patterns: tuple[re.Pattern[str], ...] = DEFAULT_WANT_TO_READ_PATTERNS,
) -> tuple[int, int]:
    """Extract "currently reading" and "want to read" reader counts.

    Args:
        soup: Parsed BeautifulSoup instance of the page.
        html_text: Raw unparsed HTML document string.
        currently_reading_patterns: Regexes identifying active reader metrics.
        want_to_read_patterns: Regexes identifying saved/to-read metrics.

    Returns:
        A tuple of `(currently_reading_count, want_to_read_count)`.
        Defaults to `(0, 0)` if counts are omitted or unparseable.
    """
    page_text = soup.get_text(" ", strip=True)
    sources = (html_text, page_text)

    currently_reading = _search_sources_for_count(sources=sources, patterns=currently_reading_patterns)
    want_to_read = _search_sources_for_count(sources=sources, patterns=want_to_read_patterns)

    return currently_reading, want_to_read


# ---------------------------------------------------------------------------
# Author-page stats (works count, followers)
# ---------------------------------------------------------------------------

DEFAULT_BOOK_AUTHORS_PATTERNS = (re.compile(r"(\d[\d,]*)\s+books?", re.IGNORECASE),)
DEFAULT_AUTHOR_FOLLOWERS_PATTERNS = (re.compile(r"(\d[\d,]*)\s+followers?", re.IGNORECASE),)


def extract_author_stats_from_author_page(
    *,
    html_text: str,
    html_parser: str = "lxml",
    book_authors_patterns: tuple[re.Pattern[str], ...] = DEFAULT_BOOK_AUTHORS_PATTERNS,
    author_followers_patterns: tuple[re.Pattern[str], ...] = DEFAULT_AUTHOR_FOLLOWERS_PATTERNS,
) -> tuple[int | None, int | None]:
    """Parse an author page's HTML for their catalog size and follower count.

    Args:
        html_text: Raw HTML string fetched from an author's Goodreads profile.
        html_parser: BeautifulSoup parser engine name to instantiate.
            Defaults to `'lxml'`.
        book_authors_patterns: Patterns to extract published work count.
        author_followers_patterns: Patterns to extract follower count.

    Returns:
        A tuple of `(number_of_books, number_of_followers)`.
        Values are None if missing or unparseable.
    """
    text = BeautifulSoup(html_text, html_parser).get_text(" ", strip=True)
    num_books = get_int(text=get_first_match(text=text, patterns=book_authors_patterns))
    num_followers = get_int(
        text=get_first_match(text=text, patterns=author_followers_patterns)
    )
    return num_books, num_followers