import pytest
from app.ingest import clean_text, chunk_text, is_valid_chunk


class TestCleanText:
    def test_removes_null_bytes(self):
        assert clean_text("hello\x00world") == "helloworld"

    def test_truncates_at_references_section(self):
        text = "Introduction\n\nSome content\n\nReferences\n\nRef 1. Author et al."
        result = clean_text(text)
        assert "Some content" in result
        assert "Ref 1" not in result

    def test_truncates_at_bibliography(self):
        text = "Body text here.\n\nBibliography\n\n[1] Smith 2020"
        result = clean_text(text)
        assert "Body text" in result
        assert "Smith 2020" not in result

    def test_fixes_hyphenated_line_breaks(self):
        assert clean_text("hyp-\nhen") == "hyphen"

    def test_removes_urls(self):
        result = clean_text("See https://example.com/paper for details.")
        assert "https://" not in result
        assert "for details" in result

    def test_collapses_excess_newlines(self):
        result = clean_text("paragraph one\n\n\n\n\nparagraph two")
        assert "\n\n\n" not in result
        assert "paragraph one" in result
        assert "paragraph two" in result


class TestIsValidChunk:
    VALID = (
        "This is a valid chunk of text with enough words and characters "
        "to pass all the validation filters without any issues. " * 2
    )

    def test_accepts_valid_chunk(self):
        assert is_valid_chunk(self.VALID)

    def test_rejects_too_short(self):
        assert not is_valid_chunk("Too short to be useful.")

    def test_rejects_too_few_words(self):
        # 100+ chars but only a handful of words
        assert not is_valid_chunk("word " * 5 + "a" * 80)

    def test_rejects_citation_heavy(self):
        # 6 citation markers — threshold is >= 5
        base = "word " * 25
        citations = " [1] [2] [3] [4] [5] [6]"
        assert not is_valid_chunk(base + citations)

    def test_rejects_low_alphanumeric_ratio(self):
        # 280 symbol chars vs 100 alphanumeric+space chars → ratio ~26%
        symbols = "!@#$%^&*()-+=[]{}|;':\",./<>?" * 10
        words = " word" * 20
        assert not is_valid_chunk(symbols + words)

    def test_rejects_very_short_average_word_length(self):
        # 100 single-letter words → avg word length = 1 < 3
        assert not is_valid_chunk("a b c d e f g h i j " * 10)

    def test_rejects_noisy_first_line_conference(self):
        noisy = "Published as a conference paper at ICLR 2024\n" + self.VALID
        assert not is_valid_chunk(noisy)

    def test_rejects_noisy_first_line_preprint(self):
        noisy = "Preprint\n" + self.VALID
        assert not is_valid_chunk(noisy)


class TestChunkText:
    def test_splits_long_text_into_multiple_chunks(self):
        text = "word " * 1000
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_short_text_stays_as_single_chunk(self):
        chunks = chunk_text("Short sentence.")
        assert len(chunks) == 1

    def test_chunks_are_strings(self):
        chunks = chunk_text("Some sample text. " * 200)
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_max_size(self):
        text = "word " * 2000
        chunks = chunk_text(text)
        # Each chunk should respect the 1500-char limit (with some tolerance for overlap)
        assert all(len(c) <= 1600 for c in chunks)
