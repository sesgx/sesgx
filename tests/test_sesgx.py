from dataclasses import dataclass

import pytest
from sesgx import sesgx


@dataclass
class MockedTopicExtractionModel(sesgx.TopicExtractionModel):
    n_words_per_topic: int

    def extract(self, docs: list[str]) -> list[list[str]]:
        return [doc.split(",")[: self.n_words_per_topic] for doc in docs]


@dataclass
class MockedWordEnrichmentModel(sesgx.WordEnrichmentModel):
    n_enrichments: int

    def enrich(self, word: str) -> list[str]:
        return [f"{word}_{i + 1}" for i in range(self.n_enrichments)]


def test_default_word_enrichment_model_should_return_empty_list():
    word_enrichment_model = sesgx.DefaultWordEnrichmentModel()

    result = word_enrichment_model.enrich("word")
    expected = []
    assert result == expected


@pytest.mark.parametrize(
    "tokens,operator,use_double_quotes,use_parenthesis,expected",
    [
        (
            ["machine", "learning", "computer"],
            "AND",
            False,
            False,
            "machine AND learning AND computer",
        ),
        (
            ["machine", "learning", "computer"],
            "OR",
            False,
            False,
            "machine OR learning OR computer",
        ),
        (
            ["machine", "learning", "computer"],
            "AND",
            True,
            False,
            '"machine" AND "learning" AND "computer"',
        ),
        (
            ["machine", "learning", "computer"],
            "OR",
            False,
            True,
            "(machine) OR (learning) OR (computer)",
        ),
        (
            ["machine", "learning", "computer"],
            "AND",
            True,
            True,
            '("machine") AND ("learning") AND ("computer")',
        ),
    ],
)
def test_join_tokens_with_operator(
    tokens,
    operator,
    use_double_quotes,
    use_parenthesis,
    expected,
):
    result = sesgx._join_tokens_with_operator(
        tokens,
        operator,
        use_double_quotes=use_double_quotes,
        use_parenthesis=use_parenthesis,
    )

    assert result == expected


def test_default_string_formulation_model_should_work_without_enriched_words():
    string_formulation_model = sesgx.DefaultStringFormulationModel()

    # 1 topic with 2 words
    result = string_formulation_model.formulate(
        [
            {"machine": [], "learning": []},
        ]
    )
    expected = '("machine" AND "learning")'
    assert result == expected

    # 2 topics with 2 words each
    result = string_formulation_model.formulate(
        [
            {"machine": [], "learning": []},
            {"computer": [], "science": []},
        ]
    )
    expected = '("machine" AND "learning") OR ("computer" AND "science")'
    assert result == expected


def test_string_formulation_model_for_enrichment_should_work_with_enriched_words():
    string_formulation_model = sesgx.StringFormulationModelForEnrichment()

    # 1 topic with 2 words and 1 enriched word
    result = string_formulation_model.formulate(
        [
            {"machine": ["machine_1"], "learning": ["learning_1"]},
        ]
    )
    expected = '(("machine" OR "machine_1") AND ("learning" OR "learning_1"))'
    assert result == expected

    # 2 topics with 2 words each and 1 enriched word each
    result = string_formulation_model.formulate(
        [
            {"machine": ["machine_1"], "learning": ["learning_1"]},
            {"computer": ["computer_1"], "science": ["science_1"]},
        ]
    )
    expected = '(("machine" OR "machine_1") AND ("learning" OR "learning_1")) OR (("computer" OR "computer_1") AND ("science" OR "science_1"))'
    assert result == expected


def test_enrich_topic_should_return_dict_mapping_a_topic_word_to_its_enriched_words():
    word_enrichment_model = MockedWordEnrichmentModel(n_enrichments=1)
    topic = ["machine", "learning"]

    enriched_topic = sesgx._enrich_topic(
        topic,
        word_enrichment_model=word_enrichment_model,
    )

    for word in topic:
        assert word in enriched_topic
        assert enriched_topic[word] == [f"{word}_1"]

    assert len(enriched_topic) == len(topic)


def test_sesg_should_use_default_word_enrichment_model_when_not_provided():
    topic_extraction_model = MockedTopicExtractionModel(n_words_per_topic=1)

    sesg = sesgx.SeSG(
        topic_extraction_model=topic_extraction_model,
    )

    assert isinstance(sesg.word_enrichment_model, sesgx.DefaultWordEnrichmentModel)


def test_sesg_should_use_default_string_formulation_model_when_not_provided():
    topic_extraction_model = MockedTopicExtractionModel(n_words_per_topic=1)

    sesg = sesgx.SeSG(
        topic_extraction_model=topic_extraction_model,
    )

    assert isinstance(
        sesg.string_formulation_model, sesgx.DefaultStringFormulationModel
    )


def test_sesg_without_word_enrichment_should_work_as_expected():
    topic_extraction_model = MockedTopicExtractionModel(n_words_per_topic=2)

    sesg = sesgx.SeSG(
        topic_extraction_model=topic_extraction_model,
    )

    result = sesg.generate(["machine,learning", "computer,science"])
    expected = '("machine" AND "learning") OR ("computer" AND "science")'

    assert result == expected


def test_sesg_with_word_enrichment_should_work_as_expected():
    topic_extraction_model = MockedTopicExtractionModel(n_words_per_topic=2)
    word_enrichment_model = MockedWordEnrichmentModel(n_enrichments=1)

    sesg = sesgx.SeSG(
        topic_extraction_model=topic_extraction_model,
        word_enrichment_model=word_enrichment_model,
        string_formulation_model=sesgx.StringFormulationModelForEnrichment(),
    )

    result = sesg.generate(["machine,learning", "computer,science"])
    expected = '(("machine" OR "machine_1") AND ("learning" OR "learning_1")) OR (("computer" OR "computer_1") AND ("science" OR "science_1"))'

    assert result == expected
