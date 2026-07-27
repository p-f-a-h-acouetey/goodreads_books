"""Pytest suite for src.goodreads_etl.utils.page_parser."""

from __future__ import annotations

from bs4 import BeautifulSoup

from src.goodreads_etl.utils.page_parser import (
    DEFAULT_CURRENTLY_READING_PATTERNS,
    extract_author_stats_from_author_page,
    extract_description,
    extract_first_author_and_url,
    extract_first_published,
    extract_format,
    extract_genres,
    extract_json_ld,
    extract_next_data,
    extract_num_pages,
    extract_num_reviews,
    extract_publisher,
    extract_reading_stats,
    extract_series,
    find_first_key,
)


def make_soup(html_str: str) -> BeautifulSoup:
    """Parse an HTML string into a BeautifulSoup tree using the lxml parser.

    Args:
        html_str: Raw HTML content string.

    Returns:
        BeautifulSoup: Parsed HTML document object model.
    """
    return BeautifulSoup(html_str, "lxml")


class TestExtractJsonLd:
    """Tests for `extract_json_ld` HTML script block extraction."""

    def test_extracts_book_type_entry(self) -> None:
        """Verify successful extraction of JSON-LD block with @type 'Book'."""
        html_str = """
        <script type="application/ld+json">
        {"@type": "Book", "name": "Dune"}
        </script>
        """
        soup = make_soup(html_str)
        result = extract_json_ld(soup=soup)
        assert result["name"] == "Dune"

    def test_returns_none_when_no_script_present(self) -> None:
        """Verify None is returned when no JSON-LD script tag exists."""
        soup = make_soup("<html><head></head></html>")
        assert extract_json_ld(soup=soup) is None

    def test_returns_none_for_invalid_json(self) -> None:
        """Verify None is returned when JSON-LD script contains malformed JSON."""
        soup = make_soup('<script type="application/ld+json">{not valid json}</script>')
        assert extract_json_ld(soup=soup) is None

    def test_skips_empty_script_body(self) -> None:
        """Verify empty script tags are safely handled and return None."""
        soup = make_soup('<script type="application/ld+json"></script>')
        assert extract_json_ld(soup=soup) is None

    def test_skips_empty_decoded_json(self) -> None:
        """Verify empty JSON object payload returns None."""
        soup = make_soup('<script type="application/ld+json">{}</script>')
        assert extract_json_ld(soup=soup) is None

    def test_top_level_scalar_json_returns_none(self) -> None:
        """Verify scalar top-level JSON data types return None."""
        html_str = '<script type="application/ld+json">"just a plain string"</script>'
        soup = make_soup(html_str)
        assert extract_json_ld(soup=soup) is None

    def test_dict_without_type_or_graph_returns_none(self) -> None:
        """Verify JSON dictionaries missing both '@type' and '@graph' return None."""
        html_str = '<script type="application/ld+json">{"foo": "bar"}</script>'
        soup = make_soup(html_str)
        assert extract_json_ld(soup=soup) is None

    def test_handles_graph_wrapper(self) -> None:
        """Verify extraction from JSON-LD payload structured under `@graph` array."""
        html_str = """
        <script type="application/ld+json">
        {"@graph": [{"@type": "WebPage"}, {"@type": "Book", "name": "Graph Book"}]}
        </script>
        """
        soup = make_soup(html_str)
        result = extract_json_ld(soup=soup)
        assert result["name"] == "Graph Book"

    def test_graph_key_present_but_not_a_list(self) -> None:
        """Verify non-list `@graph` payload returns None."""
        html_str = '<script type="application/ld+json">{"@graph": "not-a-list"}</script>'
        soup = make_soup(html_str)
        assert extract_json_ld(soup=soup) is None

    def test_graph_key_present_as_empty_list_returns_none(self) -> None:
        """Verify empty `@graph` list payload returns None."""
        html_str = '<script type="application/ld+json">{"@graph": []}</script>'
        soup = make_soup(html_str)
        assert extract_json_ld(soup=soup) is None

    def test_handles_list_of_entries(self) -> None:
        """Verify top-level JSON array is iterated to find valid book entry."""
        html_str = """
        <script type="application/ld+json">
        [{"@type": "Organization"}, {"@type": "Product", "name": "List Book"}]
        </script>
        """
        soup = make_soup(html_str)
        result = extract_json_ld(soup=soup)
        assert result["name"] == "List Book"

    def test_list_with_no_valid_entry_returns_none(self) -> None:
        """Verify top-level JSON list with no recognized target type returns None."""
        html_str = """
        <script type="application/ld+json">
        [{"@type": "Organization"}, {"@type": "WebPage"}]
        </script>
        """
        soup = make_soup(html_str)
        assert extract_json_ld(soup=soup) is None

    def test_first_block_invalid_second_block_valid(self) -> None:
        """Verify iteration over multiple JSON-LD script blocks until valid target is found."""
        html_str = """
        <script type="application/ld+json">{"@type": "WebPage"}</script>
        <script type="application/ld+json">{"@type": "Book", "name": "Second Block Book"}</script>
        """
        soup = make_soup(html_str)
        result = extract_json_ld(soup=soup)
        assert result["name"] == "Second Block Book"


class TestExtractNextData:
    """Tests for `extract_next_data` extracting Next.js hydration payload."""

    def test_extracts_valid_payload(self) -> None:
        """Verify successful parsing of `__NEXT_DATA__` script tag."""
        html_str = '<script id="__NEXT_DATA__">{"props": {"publisher": "Ace Books"}}</script>'
        soup = make_soup(html_str)
        assert extract_next_data(soup=soup) == {"props": {"publisher": "Ace Books"}}

    def test_returns_none_when_missing(self) -> None:
        """Verify None is returned when `__NEXT_DATA__` script tag is absent."""
        soup = make_soup("<html></html>")
        assert extract_next_data(soup=soup) is None

    def test_returns_none_when_script_body_empty(self) -> None:
        """Verify None is returned when script body is empty."""
        html_str = '<script id="__NEXT_DATA__"></script>'
        soup = make_soup(html_str)
        assert extract_next_data(soup=soup) is None

    def test_returns_none_for_invalid_json(self) -> None:
        """Verify None is returned when script contains invalid JSON syntax."""
        html_str = '<script id="__NEXT_DATA__">{invalid}</script>'
        soup = make_soup(html_str)
        assert extract_next_data(soup=soup) is None

    def test_multiple_scripts_only_matching_id_used(self) -> None:
        """Verify script search strictly targets `id="__NEXT_DATA__"`."""
        html_str = (
            '<script id="OTHER">{"a": 1}</script>'
            '<script id="__NEXT_DATA__">{"b": 2}</script>'
        )
        soup = make_soup(html_str)
        assert extract_next_data(soup=soup) == {"b": 2}


class TestFindFirstKey:
    """Tests for `find_first_key` recursive key search in nested structures."""

    def test_finds_key_in_flat_dict(self) -> None:
        """Verify key retrieval in single-level dictionary."""
        assert find_first_key(data={"publisher": "Ace Books"}, keys=frozenset({"publisher"})) == "Ace Books"

    def test_finds_key_nested_in_dict(self) -> None:
        """Verify key retrieval deeply nested inside dictionary hierarchy."""
        data = {"a": {"b": {"publisherName": "Deep Press"}}}
        assert find_first_key(data=data, keys=frozenset({"publisherName"})) == "Deep Press"

    def test_finds_key_inside_list_and_nested_lists(self) -> None:
        """Verify key retrieval nested inside arbitrary lists and dictionaries."""
        data = [{"a": [{"b": [{"imprint": "Very Deep"}]}]}]
        assert find_first_key(data=data, keys=frozenset({"imprint"})) == "Very Deep"

    def test_returns_none_when_key_absent(self) -> None:
        """Verify None is returned when requested keys are nowhere in structure."""
        assert find_first_key(data={"a": 1}, keys=frozenset({"missing"})) is None

    def test_ignores_empty_string_and_non_string_values(self) -> None:
        """Verify non-string and empty string values are skipped during search."""
        assert find_first_key(data={"publisher": ""}, keys=frozenset({"publisher"})) is None
        assert find_first_key(data={"publisher": 123}, keys=frozenset({"publisher"})) is None

    def test_handles_scalar_and_list_of_scalars_gracefully(self) -> None:
        """Verify non-dictionary scalar values do not crash recursive traversal."""
        assert find_first_key(data="just a string", keys=frozenset({"publisher"})) is None
        assert find_first_key(data=[1, 2, "three"], keys=frozenset({"publisher"})) is None


class TestExtractDescription:
    """Tests for `extract_description` html text extraction."""

    def test_extracts_from_formatted_span(self) -> None:
        """Verify description extraction from primary `.Formatted` span tag."""
        soup = make_soup('<span class="Formatted">A great book about sand.</span>')
        assert extract_description(soup=soup) == "a great book about sand."

    def test_extracts_from_data_testid_div_when_span_missing(self) -> None:
        """Verify description fallback to `data-testid="description"` div tag."""
        soup = make_soup('<div data-testid="description">Fallback description text.</div>')
        assert extract_description(soup=soup) == "fallback description text."

    def test_falls_back_to_meta_tag(self) -> None:
        """Verify description fallback to meta description HTML tag."""
        soup = make_soup('<meta itemprop="description" content="Meta description text">')
        assert extract_description(soup=soup) == "meta description text"

    def test_returns_none_when_nothing_found(self) -> None:
        """Verify None is returned when no description tags exist."""
        soup = make_soup("<html></html>")
        assert extract_description(soup=soup) is None

    def test_meta_tag_present_but_no_content_attr(self) -> None:
        """Verify meta description tag missing 'content' attribute returns None."""
        soup = make_soup('<meta itemprop="description">')
        assert extract_description(soup=soup) is None


class TestExtractGenres:
    """Tests for `extract_genres` genre tag aggregator."""

    def test_extracts_and_deduplicates_genres(self) -> None:
        """Verify unique genre list extraction formatted as comma-separated string."""
        html_str = (
            '<a href="/genres/fiction">Fiction</a>'
            '<a href="/genres/scifi">Sci-fi</a>'
            '<a href="/genres/fiction">Fiction</a>'
        )
        soup = make_soup(html_str)
        assert extract_genres(soup=soup) == "fiction, sci-fi"

    def test_returns_none_when_no_genre_links(self) -> None:
        """Verify None is returned when page lacks genre hyperlink elements."""
        soup = make_soup('<a href="/book/show/1">Not a genre</a>')
        assert extract_genres(soup=soup) is None

    def test_skips_genre_link_with_empty_text(self) -> None:
        """Verify empty genre hyperlinks are ignored."""
        html_str = '<a href="/genres/fiction"></a><a href="/genres/scifi">Sci-fi</a>'
        soup = make_soup(html_str)
        assert extract_genres(soup=soup) == "sci-fi"

    def test_ignores_links_without_href(self) -> None:
        """Verify hyperlink tags without 'href' attributes are safely ignored."""
        soup = make_soup("<a>No href here</a>")
        assert extract_genres(soup=soup) is None


class TestExtractFirstAuthorAndUrl:
    """Tests for `extract_first_author_and_url` author information parser."""

    def test_extracts_from_author_link(self) -> None:
        """Verify author name and absolute URL extraction from HTML anchor tag."""
        soup = make_soup('<a href="/author/show/1">Frank Herbert</a>')
        name, url = extract_first_author_and_url(soup=soup, json_ld=None)
        assert name == "frank herbert"
        assert url == "https://www.goodreads.com/author/show/1"

    def test_absolute_url_preserved_as_is(self) -> None:
        """Verify absolute author URL is preserved without double-prepending host domain."""
        soup = make_soup('<a href="https://www.goodreads.com/author/show/9">Some Author</a>')
        _, url = extract_first_author_and_url(soup=soup, json_ld=None)
        assert url == "https://www.goodreads.com/author/show/9"

    def test_skips_non_author_link_before_finding_author_link(self) -> None:
        """Verify non-author links are skipped until matching author URL pattern is found."""
        html_str = (
            '<a href="/book/show/1">Not an author link</a>'
            '<a href="/author/show/2">Real Author</a>'
        )
        soup = make_soup(html_str)
        name, url = extract_first_author_and_url(soup=soup, json_ld=None)
        assert name == "real author"
        assert url == "https://www.goodreads.com/author/show/2"

    def test_prefers_link_over_json_ld_when_both_present(self) -> None:
        """Verify HTML anchor tag takes precedence over JSON-LD fallback for author info."""
        soup = make_soup('<a href="/author/show/1">Link Author</a>')
        json_ld = {"author": [{"name": "JSON Author"}]}
        name, url = extract_first_author_and_url(soup=soup, json_ld=json_ld)
        assert name == "link author"
        assert url == "https://www.goodreads.com/author/show/1"

    def test_author_link_with_empty_text_falls_through_to_json_ld(self) -> None:
        """Verify author anchor tag with empty label text falls back to JSON-LD metadata."""
        soup = make_soup('<a href="/author/show/1"></a>')
        json_ld = {"author": [{"name": "Fallback Author"}]}
        name, url = extract_first_author_and_url(soup=soup, json_ld=json_ld)
        assert name == "fallback author"
        assert url is None

    def test_falls_back_to_json_ld_name_only(self) -> None:
        """Verify fallback to JSON-LD author list when HTML author links are absent."""
        soup = make_soup("<html></html>")
        json_ld = {"author": [{"name": "Frank Herbert"}]}
        name, url = extract_first_author_and_url(soup=soup, json_ld=json_ld)
        assert name == "frank herbert"
        assert url is None

    def test_json_ld_author_as_single_dict_not_list(self) -> None:
        """Verify handling of JSON-LD author represented as dictionary rather than array."""
        soup = make_soup("<html></html>")
        json_ld = {"author": {"name": "Solo Author"}}
        name, url = extract_first_author_and_url(soup=soup, json_ld=json_ld)
        assert name == "solo author"
        assert url is None

    def test_json_ld_author_list_skips_non_dict_and_nameless_entries(self) -> None:
        """Verify robust handling of invalid/incomplete author dicts in JSON-LD list."""
        soup = make_soup("<html></html>")
        json_ld = {"author": ["not-a-dict", {"role": "editor"}, {"name": "Valid Author"}]}
        name, url = extract_first_author_and_url(soup=soup, json_ld=json_ld)
        assert name == "valid author"
        assert url is None

    def test_returns_none_none_when_nothing_found(self) -> None:
        """Verify (None, None) returned when no author info exists in HTML or JSON-LD."""
        soup = make_soup("<html></html>")
        assert extract_first_author_and_url(soup=soup, json_ld=None) == (None, None)

    def test_json_ld_author_list_with_no_name_returns_none_none(self) -> None:
        """Verify (None, None) returned when JSON-LD author entries lack 'name' key."""
        soup = make_soup("<html></html>")
        json_ld = {"author": [{"foo": "bar"}]}
        assert extract_first_author_and_url(soup=soup, json_ld=json_ld) == (None, None)


class TestExtractNumPages:
    """Tests for `extract_num_pages` page length parser."""

    def test_prefers_json_ld(self) -> None:
        """Verify `numberOfPages` from JSON-LD takes precedence."""
        soup = make_soup("<html></html>")
        assert extract_num_pages(soup=soup, json_ld={"numberOfPages": 412}) == 412

    def test_json_ld_present_but_zero_falls_back_to_page_text(self) -> None:
        """Verify fallback to page text when JSON-LD page count is zero."""
        soup = make_soup("<p>350 pages</p>")
        assert extract_num_pages(soup=soup, json_ld={"numberOfPages": 0}) == 350

    def test_falls_back_to_page_text(self) -> None:
        """Verify regex extraction of page count from surrounding body text."""
        soup = make_soup("<p>This edition has 350 pages total</p>")
        assert extract_num_pages(soup=soup, json_ld=None) == 350

    def test_falls_back_to_page_text_with_label_pattern(self) -> None:
        """Verify regex extraction of page count using key-value string patterns."""
        soup = make_soup("<p>Pages: 500</p>")
        assert extract_num_pages(soup=soup, json_ld=None) == 500

    def test_returns_none_when_absent(self) -> None:
        """Verify None is returned when page count cannot be determined."""
        soup = make_soup("<p>no page info here</p>")
        assert extract_num_pages(soup=soup, json_ld=None) is None


class TestExtractFormat:
    """Tests for `extract_format` binding/publication format parser."""

    def test_detects_known_format_keyword(self) -> None:
        """Verify identification of standard format keywords (e.g. Paperback)."""
        soup = make_soup("<p>Format: Paperback, 300 pages</p>")
        assert extract_format(soup=soup) == "paperback"

    def test_detects_kindle_edition_multiword_format(self) -> None:
        """Verify identification of multi-word format patterns like 'Kindle Edition'."""
        soup = make_soup("<p>Kindle Edition, 300 pages</p>")
        assert extract_format(soup=soup) == "kindle edition"

    def test_returns_none_when_no_format_present(self) -> None:
        """Verify None is returned when no format indicator is detected."""
        soup = make_soup("<p>nothing relevant here</p>")
        assert extract_format(soup=soup) is None


class TestExtractSeries:
    """Tests for `extract_series` book series name parser."""

    def test_extracts_series_from_hash_notation(self) -> None:
        """Verify series extraction from title string matching '#N' notation."""
        soup = make_soup('<a href="/series/1">Dune (Dune #1)</a>')
        assert extract_series(soup=soup) == "dune"

    def test_extracts_series_from_series_colon_notation(self) -> None:
        """Verify series extraction from title string matching 'Series: ...' notation."""
        soup = make_soup('<a href="/series/1">Series: The Chronicles</a>')
        assert extract_series(soup=soup) == "the chronicles"

    def test_returns_none_when_no_series_link(self) -> None:
        """Verify None is returned when no series hyperlink exists."""
        soup = make_soup('<a href="/book/show/1">Standalone Book</a>')
        assert extract_series(soup=soup) is None

    def test_series_link_with_empty_text_is_skipped(self) -> None:
        """Verify series hyperlinks with empty label text are skipped."""
        soup = make_soup('<a href="/series/1"></a>')
        assert extract_series(soup=soup) is None

    def test_series_link_with_no_pattern_match_returns_none(self) -> None:
        """Verify series hyperlinks that fail naming pattern match return None."""
        soup = make_soup('<a href="/series/1">Just a plain title</a>')
        assert extract_series(soup=soup) is None

    def test_first_series_link_without_match_falls_to_second(self) -> None:
        """Verify iteration through multiple series links until pattern matches."""
        html_str = (
            '<a href="/series/1">No match here</a>'
            '<a href="/series/2">Dune (Dune #1)</a>'
        )
        soup = make_soup(html_str)
        assert extract_series(soup=soup) == "dune"


class TestExtractNumReviews:
    """Tests for `extract_num_reviews` aggregate review count parser."""

    def test_prefers_json_ld_aggregate_rating(self) -> None:
        """Verify preference for review count inside JSON-LD `aggregateRating`."""
        soup = make_soup("<html></html>")
        json_ld = {"aggregateRating": {"reviewCount": 5000}}
        assert extract_num_reviews(soup=soup, json_ld=json_ld) == 5000

    def test_json_ld_present_without_aggregate_rating_key_falls_back(self) -> None:
        """Verify fallback to page text when JSON-LD lacks `aggregateRating`."""
        soup = make_soup("<p>1,234 reviews</p>")
        json_ld = {"name": "Some Book"}
        assert extract_num_reviews(soup=soup, json_ld=json_ld) == 1234

    def test_json_ld_aggregate_rating_none_falls_back(self) -> None:
        """Verify fallback to page text when `aggregateRating` key is None."""
        soup = make_soup("<p>1,234 reviews</p>")
        json_ld = {"aggregateRating": None}
        assert extract_num_reviews(soup=soup, json_ld=json_ld) == 1234

    def test_falls_back_to_page_text(self) -> None:
        """Verify regex extraction of review count from body text."""
        soup = make_soup("<p>1,234 reviews</p>")
        assert extract_num_reviews(soup=soup, json_ld=None) == 1234

    def test_falls_back_to_page_text_label_pattern(self) -> None:
        """Verify regex extraction of review count using label-value pattern."""
        soup = make_soup("<p>Reviews: 42</p>")
        assert extract_num_reviews(soup=soup, json_ld=None) == 42

    def test_returns_none_when_absent(self) -> None:
        """Verify None is returned when review count cannot be located."""
        soup = make_soup("<p>nothing here</p>")
        assert extract_num_reviews(soup=soup, json_ld=None) is None


class TestExtractFirstPublished:
    """Tests for `extract_first_published` initial publication date parser."""

    def test_prefers_page_text_first_published_phrase(self) -> None:
        """Verify preference for explicit 'First published ...' text phrase."""
        soup = make_soup("<p>First published August 1, 1965</p>")
        assert extract_first_published(soup=soup, json_ld=None) == "august 1, 1965"

    def test_page_text_generic_published_phrase(self) -> None:
        """Verify extraction from generic 'Published ...' text phrase."""
        soup = make_soup("<p>Published March 3, 2001</p>")
        assert extract_first_published(soup=soup, json_ld=None) == "march 3, 2001"

    def test_falls_back_to_json_ld_date_published(self) -> None:
        """Verify fallback to `datePublished` field in JSON-LD."""
        soup = make_soup("<p>no relevant text</p>")
        json_ld = {"datePublished": "1965-08-01"}
        assert extract_first_published(soup=soup, json_ld=json_ld) == "1965-08-01"

    def test_page_text_pattern_present_but_empty_capture_falls_to_json_ld(self) -> None:
        """Verify empty regex capture in text falls back to JSON-LD date."""
        soup = make_soup("<p>First published </p>")
        json_ld = {"datePublished": "1965-08-01"}
        assert extract_first_published(soup=soup, json_ld=json_ld) in ("1965-08-01", None)

    def test_json_ld_present_but_date_published_empty_returns_none(self) -> None:
        """Verify empty string `datePublished` in JSON-LD yields None."""
        soup = make_soup("<p>no dates here</p>")
        json_ld = {"datePublished": ""}
        assert extract_first_published(soup=soup, json_ld=json_ld) is None

    def test_returns_none_when_absent(self) -> None:
        """Verify None is returned when publication date cannot be found."""
        soup = make_soup("<p>no dates here</p>")
        assert extract_first_published(soup=soup, json_ld=None) is None


class TestExtractPublisher:
    """Tests for `extract_publisher` publishing entity parser."""

    def test_prefers_next_data(self) -> None:
        """Verify preference for publisher metadata extracted from Next.js payload."""
        soup = make_soup("<html></html>")
        next_data = {"publisher": "Next Data Press"}
        assert extract_publisher(soup=soup, json_ld=None, next_data=next_data) == "next data press"

    def test_next_data_present_but_no_match_falls_to_json_ld(self) -> None:
        """Verify fallback to JSON-LD when Next.js payload lacks publisher key."""
        soup = make_soup("<html></html>")
        next_data = {"unrelated": "value"}
        json_ld = {"publisher": {"name": "JSON LD Press"}}
        result = extract_publisher(soup=soup, json_ld=json_ld, next_data=next_data)
        assert result == "json ld press"

    def test_falls_back_to_json_ld_dict_form(self) -> None:
        """Verify publisher extraction when JSON-LD represents publisher as nested object."""
        soup = make_soup("<html></html>")
        json_ld = {"publisher": {"name": "JSON LD Press"}}
        assert extract_publisher(soup=soup, json_ld=json_ld, next_data=None) == "json ld press"

    def test_falls_back_to_json_ld_string_form(self) -> None:
        """Verify publisher extraction when JSON-LD represents publisher as plain string."""
        soup = make_soup("<html></html>")
        json_ld = {"publisher": "String Press"}
        assert extract_publisher(soup=soup, json_ld=json_ld, next_data=None) == "string press"

    def test_json_ld_publisher_wrong_type_falls_back_to_page_text(self) -> None:
        """Verify fallback to body text when JSON-LD publisher field is invalid type."""
        soup = make_soup("<p>Publisher: Page Text Press</p>")
        json_ld = {"publisher": 12345}
        result = extract_publisher(soup=soup, json_ld=json_ld, next_data=None)
        assert result == "page text press"

    def test_falls_back_to_page_text_when_next_data_and_json_ld_absent(self) -> None:
        """Verify fallback to HTML body text pattern matching when structured data is missing."""
        soup = make_soup("<p>Publisher: Only Page Text</p>")
        assert extract_publisher(soup=soup, json_ld=None, next_data=None) == "only page text"

    def test_returns_none_when_absent_everywhere(self) -> None:
        """Verify None is returned when publisher cannot be determined from any source."""
        soup = make_soup("<p>nothing relevant</p>")
        assert extract_publisher(soup=soup, json_ld=None, next_data=None) is None


class TestExtractReadingStats:
    """Tests for `extract_reading_stats` social status counts parser."""

    def test_extracts_want_to_read_count(self) -> None:
        """Verify extraction of 'want to read' user metric."""
        html_str = "<p>500 people want to read this</p>"
        soup = make_soup(html_str)
        _, want_to_read = extract_reading_stats(soup=soup, html_text=html_str)
        assert want_to_read == 500

    def test_returns_zero_zero_when_nothing_found(self) -> None:
        """Verify default tuple (0, 0) is returned when reading stats are absent."""
        html_str = "<p>no stats here</p>"
        soup = make_soup(html_str)
        assert extract_reading_stats(soup=soup, html_text=html_str) == (0, 0)

    def test_finds_count_in_html_text_when_absent_from_rendered_page_text(self) -> None:
        """Verify fallback to raw HTML string matching when target text is inside script tags."""
        html_str = '<script>var data = "1000 people want to read";</script>'
        soup = make_soup("<p>rendered text has nothing</p>")
        _, want_to_read = extract_reading_stats(soup=soup, html_text=html_str)
        assert want_to_read == 1000

    def test_currently_reading_pattern_has_broken_literal_underscore(self) -> None:
        """Document and verify known regex issue where '\\_' prevents space-matching.

        Note:
            DEFAULT_CURRENTLY_READING_PATTERNS contains a literal escaped underscore
            'currently\\_reading' rather than a space, causing standard Goodreads page
            content ('currently reading') to evaluate to 0.
        """
        html_str = "<p>1,234 people are currently reading this book</p>"
        soup = make_soup(html_str)
        currently_reading, _ = extract_reading_stats(soup=soup, html_text=html_str)
        assert currently_reading == 0

    def test_pattern_matches_literal_underscore_variant(self) -> None:
        """Verify current regex matches literal underscore variant 'currently_reading'."""
        html_str = "<p>1,234 people are currently_reading this book</p>"
        soup = make_soup(html_str)
        currently_reading, _ = extract_reading_stats(soup=soup, html_text=html_str)
        assert currently_reading == 1234
        assert DEFAULT_CURRENTLY_READING_PATTERNS[0].search(html_str) is not None


class TestExtractAuthorStatsFromAuthorPage:
    """Tests for `extract_author_stats_from_author_page` author profile parser."""

    def test_extracts_book_and_follower_counts(self) -> None:
        """Verify extraction of total published books count and follower count."""
        html_str = "<html><body><p>42 books, 10,000 followers</p></body></html>"
        num_books, num_followers = extract_author_stats_from_author_page(html_text=html_str)
        assert num_books == 42
        assert num_followers == 10000

    def test_returns_none_none_when_absent(self) -> None:
        """Verify (None, None) is returned when author profile metrics are missing."""
        html_str = "<html><body><p>nothing relevant</p></body></html>"
        num_books, num_followers = extract_author_stats_from_author_page(html_text=html_str)
        assert num_books is None
        assert num_followers is None

    def test_only_books_present_followers_none(self) -> None:
        """Verify partial extraction when only book count metric exists on profile."""
        html_str = "<html><body><p>15 books</p></body></html>"
        num_books, num_followers = extract_author_stats_from_author_page(html_text=html_str)
        assert num_books == 15
        assert num_followers is None

    def test_only_followers_present_books_none(self) -> None:
        """Verify partial extraction when only follower count metric exists on profile."""
        html_str = "<html><body><p>2,500 followers</p></body></html>"
        num_books, num_followers = extract_author_stats_from_author_page(html_text=html_str)
        assert num_books is None
        assert num_followers == 2500