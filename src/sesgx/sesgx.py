from typing import Dict, Iterable, List, Literal, Protocol


class TopicExtractionModel(Protocol):
    """Interface for a topic extraction model that can be used with SeSG.

    Must have a method `extract` that receives a list of documents and returns a list of topics.
    """

    def extract(self, docs: List[str]) -> List[List[str]]: ...


class WordEnrichmentModel(Protocol):
    """Interface for a word enrichment model that can be used with SeSG.

    Must have a method `enrich` that receives a word and returns a list of enriched words.
    """

    def enrich(self, word: str) -> List[str]: ...


class StringFormulationModel(Protocol):
    """ "Interface for a string formulation model that can be used with SeSG.

    Must have a method `formulate` that receives the data for formulation and returns a string.

    The data is represented as a list of dictionaries, where each dictionary represents a topic and has the original words as keys and the list of enriched words as values.
    """

    def formulate(self, data: List[Dict[str, List[str]]]) -> str: ...


class DefaultWordEnrichmentModel(WordEnrichmentModel):
    """Default word enrichment model. Does not perform the enrichment, i.e., returns an empty list."""

    def enrich(self, word: str) -> List[str]:
        return []


def _join_tokens_with_operator(
    tokens: Iterable[str],
    operator: Literal["AND", "OR"],
    *,
    use_double_quotes: bool = False,
    use_parenthesis: bool = False,
) -> str:
    """Joins the tokens in the list using the provided operator.

    First checks if should surround with double quotes, then checks if should surround with parenthesis.
    If both are set to True, will add both double quotes and parenthesis.

    Args:
        operator (Literal["AND", "OR"]): Operator to use to join.
        tokens (Iterable[str]): Tokens to join.
        use_double_quotes (Optional[bool]): Whether to put double quotes surrounding each token.
        use_parenthesis (Optional[bool]): Whether to put parenthesis surrounding each token.

    Returns:
        A string with the joined tokens.

    Examples:
        >>> _join_tokens_with_operator(["machine", "learning", "SLR"], "AND", use_double_quotes=True)
        '"machine" AND "learning" AND "SLR"'
    """  # noqa: E501
    if use_double_quotes:
        tokens = (f'"{token}"' for token in tokens)

    if use_parenthesis:
        tokens = (f"({token})" for token in tokens)

    return f" {operator} ".join(tokens)


class StringFormulationModelForEnrichment(StringFormulationModel):
    """String formulation model that formulates a search string using topics that were enriched.

    The rules are:

    1. Enriched words are joined with "OR" since they are synonyms.
    1. Terms of topics are joined with "AND" since they are related to the same topic and describe a document (or a set of documents).
    1. Topics are joined with "OR" since each topic describe a set of documents and a document can be described by multiple topics.
    """

    def formulate(self, data: List[Dict[str, List[str]]]) -> str:
        topics_part: List[str] = []

        for topic in data:
            enriched_words_part: List[str] = []

            for word, enriched_words in topic.items():
                # enriched words are joined with "OR"
                # since they are *synonyms
                # *not necessarily synonyms, but words that are related to the original word
                s = _join_tokens_with_operator(
                    [word, *enriched_words],
                    "OR",
                    use_double_quotes=True,
                )

                enriched_words_part.append(s)

            # terms of topics (in this case, sets of enriched words)
            # are joined with "AND"
            # since they are related to the same topic
            # and we assume that a topic describes a document
            s = _join_tokens_with_operator(
                enriched_words_part,
                "AND",
                use_parenthesis=True,
            )

            topics_part.append(s)

        # topics are joined with "OR"
        # since a document can be related to multiple topics
        # and also, since a topic describes a document,
        # we want to find all relevand documents as an UNION
        # hence, we use "OR"
        string = _join_tokens_with_operator(
            topics_part,
            "OR",
            use_parenthesis=True,
        )

        return string


class DefaultStringFormulationModel(StringFormulationModel):
    """String formulation model that formulates a search string using topics that were not enriched.

    The rules are:

    1. Words from the same topic are joined with AND
    1. Topics are joined with "OR" since each topic describe a set of documents and a document can be described by multiple topics.
    """

    def formulate(self, data: List[Dict[str, List[str]]]) -> str:
        topics_part: List[str] = []

        for topic in data:
            topic_words = list(topic.keys())
            s = _join_tokens_with_operator(
                topic_words,
                "AND",
                use_double_quotes=True,
            )

            topics_part.append(s)

        # topics are joined with "OR"
        string = _join_tokens_with_operator(
            topics_part,
            "OR",
            use_parenthesis=True,
        )

        return string


def _enrich_topic(
    topic: List[str],
    word_enrichment_model: WordEnrichmentModel,
) -> Dict[str, List[str]]:
    """Enriches a topic using the provided word enrichment model.

    Args:
        topic (List[str]): Topic to enrich.
        word_enrichment_model (WordEnrichmentModel): Word enrichment model to use.

    Returns:
        A dictionary with the original words as keys and the list of enriched words as values.
    """
    return {word: word_enrichment_model.enrich(word) for word in topic}


class SeSG:
    """Search String Generator (SeSG) framework."""

    topic_extraction_model: TopicExtractionModel
    word_enrichment_model: WordEnrichmentModel
    string_formulation_model: StringFormulationModel

    def __init__(
        self,
        topic_extraction_model: TopicExtractionModel,
        word_enrichment_model: WordEnrichmentModel | None = None,
        string_formulation_model: StringFormulationModel | None = None,
    ):
        """Initializes the SeSG framework.

        - If `word_enrichment_model` is not provided, will not perform word enrichment.
        - If `string_formulation_model` is not provided, will use the default string formulation model which only works with topics that were not enriched.
        """
        self.topic_extraction_model = topic_extraction_model
        self.word_enrichment_model = (
            word_enrichment_model or DefaultWordEnrichmentModel()
        )
        self.string_formulation_model = (
            string_formulation_model or DefaultStringFormulationModel()
        )

    def generate(self, docs: List[str]) -> str:
        """Generates a search string using the provided models.

        Args:
            docs (List[str]): List of documents, where each document represents a relevant study. The document can include, for example, the title, abstract, and keywords of the study.

        Returns:
            The search string.
        """
        topics = self.topic_extraction_model.extract(docs)
        enriched_topics = [
            _enrich_topic(topic, self.word_enrichment_model) for topic in topics
        ]

        search_string = self.string_formulation_model.formulate(enriched_topics)

        return search_string
