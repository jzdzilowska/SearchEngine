import pytest
import math
from index import Indexer

# Here's an example test case to make sure your tests are working
# Remember that all test functions must start with "test"
def test_example():
    assert 2 == 1 + 1

def test_process_document():
    # Tests on just inputs; regardless of passed-in wiki 
    indexer = Indexer("SmallWiki.xml", "Title", "Doc", "Words")
    title = "Test"
    id = 1
    body = "Here are the stop words. And, like, the. And [[Example page|(See this page)]]."
    expected = {'test', 'stop', 'word', 'like', 'see', 'page'}
    assert set(indexer.process_document(title, id, body)) == expected

    # doc with no stop words or links
    title2 = "Another File"
    id2 = 2
    body2 = "Wow another test document without any stop words or links"
    expected2= {'wow', 'anoth', 'file', 'test', 'document', 'without', 'stop', 'word', 'link'}
    assert set(indexer.process_document(title2, id2, body2)) == expected2

    # only stop words 
    title3 = "and"
    id3 = 3
    body3 = "This is a with and, the, in, of, with."
    expected3 = []
    assert indexer.process_document(title3, id3, body3) == expected3

    # only title 
    title4 = "Another File"
    id4 = 4
    body4 = ""
    expected4= {'anoth', 'file'}
    assert set(indexer.process_document(title4, id4, body4)) == expected4


def test_compute_tf():
    example_index = Indexer("wikis/ExampleWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index.parse()
    tf = example_index.compute_tf()
    assert tf["miniatur"][1] == 1 
    assert tf["wiki"][1] == float(1/3)
    assert tf["schnauzer"][1] == float(2/3)
    assert tf["schnauzer"][2] == float(1/4)
    assert tf["link"][1] == float(1/3) 
    assert tf["page"][1] == float(1/3) 
    assert tf["page"][2] == float(1/2)
    assert tf["pomeranian"][1] == float(1/3)
    assert tf["pomeranian"][2] == 1 
    assert len(tf) == 6

def test_compute_idf():
    example_index = Indexer("wikis/ExampleWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index.parse()
    idf = example_index.compute_idf()
    assert idf["miniatur"] == math.log(2)
    assert idf["wiki"] == math.log(2)
    assert idf["schnauzer"] == 0
    assert idf["link"] == math.log(2)
    assert idf["page"] == 0
    assert idf["pomeranian"] == 0 
    assert len(idf) == 6

def test_compute_term_relevance():
    example_index = Indexer("wikis/ExampleWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index.parse()
    term_revelance = example_index.compute_term_relevance()
    assert term_revelance["miniatur"][1] == math.log(2)
    assert term_revelance["wiki"][1] == math.log(2) * float(1/3)
    assert term_revelance["schnauzer"][1] == 0
    assert term_revelance["schnauzer"][2] == 0
    assert term_revelance["link"][1] == math.log(2) * float(1/3)
    assert term_revelance["page"][1] == 0
    assert term_revelance["page"][2] == 0
    assert term_revelance["pomeranian"][1] == 0 
    assert term_revelance["pomeranian"][2] == 0 
    assert len(term_revelance) == 6

def test_pagerank(): 
    example_index = Indexer("wikis/PageRankWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index.parse()
    pagerank = example_index.compute_page_rank()
    rank_sum = sum(pagerank.values())
    assert rank_sum == pytest.approx(1.0)

    example_index2 = Indexer("wikis/PageRankExample1.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index2.parse()
    pagerank2 = example_index2.compute_page_rank()
    rank_sum2 = sum(pagerank2.values())
    assert rank_sum2 == pytest.approx(1.0)
    assert pagerank2[1] == pytest.approx(0.4326, abs=0.0001)
    assert pagerank2[2] == pytest.approx(0.234, abs=0.0001)
    assert pagerank2[3] == pytest.approx(0.3333, abs=0.0001)

    example_index3 = Indexer("wikis/PageRankExample2.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index3.parse()
    pagerank3 = example_index3.compute_page_rank()
    rank_sum3 = sum(pagerank3.values())
    assert rank_sum3 == pytest.approx(1.0)
    assert pagerank3[1] == pytest.approx(0.2018, abs=0.0001)
    assert pagerank3[2] == pytest.approx(0.0375, abs=0.0001)
    assert pagerank3[3] == pytest.approx(0.3740, abs=0.0001)
    assert pagerank3[4] == pytest.approx(0.3867, abs=0.0001)

    example_index4 = Indexer("wikis/IroningWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index4.parse()
    pagerank4 = example_index4.compute_page_rank()
    rank_sum4 = sum(pagerank4.values())
    assert rank_sum4 == pytest.approx(1.0)
    assert pagerank4[1] == pytest.approx(0.3333, abs=0.001)
    assert pagerank4[2] == pytest.approx(0.43264, abs=0.001)
    assert pagerank4[3] == pytest.approx(0.2340, abs=0.001)

    example_index5 = Indexer("wikis/OneLinkWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index5.parse()
    pagerank5 = example_index5.compute_page_rank()
    rank_sum5 = sum(pagerank5.values())
    assert rank_sum5 == pytest.approx(1.0)
    assert pagerank5[10] == pytest.approx(0.5, abs=0.001)
    assert pagerank5[30] == pytest.approx(0.5, abs=0.001)

    example_index6 = Indexer("wikis/OutsideLinksWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index6.parse()
    pagerank6 = example_index6.compute_page_rank()
    rank_sum6 = sum(pagerank6.values())
    assert rank_sum6 == pytest.approx(1.0)
    assert pagerank6[10] == pytest.approx(0.333, abs=0.001)
    assert pagerank6[20] == pytest.approx(0.333, abs=0.001)
    assert pagerank6[30] == pytest.approx(0.333, abs=0.001)

    example_index7 = Indexer("wikis/MixedLinksWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index7.parse()
    pagerank7 = example_index7.compute_page_rank()
    rank_sum7 = sum(pagerank7.values())
    assert rank_sum7 == pytest.approx(1.0)
    assert pagerank7[10] == pytest.approx(0.271, abs=0.001)
    assert pagerank7[20] == pytest.approx(0.327, abs=0.001)
    assert pagerank7[30] == pytest.approx(0.190, abs=0.001)
    assert pagerank7[44] == pytest.approx(0.211, abs=0.001)

def file_as_set(filename):
    """
    Returns all of the non-empty lines in the file, as strings in a set.
    """
    line_set = set()
    with open(filename, "r") as file:
        line = file.readline()
        while line and len(line.strip()) > 0:
            line_set.add(line.strip())
            line = file.readline()
    return line_set

def test_file_contents():
    simple_index = Indexer("wikis/SimpleWiki.xml", "simple_titles.txt",
       "simple_docs.txt", "simple_words.txt")
    simple_index.run() # run the indexer to write to the files
    titles_contents = file_as_set("simple_titles.txt")
    assert len(titles_contents) == 2
    assert "200::Example page" in titles_contents
    assert "30::Page with links" in titles_contents

    example_index = Indexer("wikis/ExampleWiki.xml", "example_titles.txt",
    "example_docs.txt", "example_words.txt")
    example_index.run() # run the indexer to write to the files
    titles_contents = file_as_set("example_titles.txt")
    assert len(titles_contents) == 2
    assert "1::Miniature Schnauzers" in titles_contents
    assert "2::Pomeranians" in titles_contents

