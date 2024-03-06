# sesgx

> SeSG (Search String Generator) is a framework to help automate search string generation for SLRs.

## Usage

```python
from dataclasses import dataclass

from sesgx import SeSG, TopicExtractionModel


@dataclass
class DummyTopicExtractionModel(sesgx.TopicExtractionModel):
    n_words_per_topic: int

    def extract(self, docs: list[str]) -> list[list[str]]:
        return [doc.split(",")[: self.n_words_per_topic] for doc in docs]


topic_extraction_model = DummyTopicExtractionModel(
    n_words_per_topic=2,
)

sesg = SeSG(
    topic_extraction_model=topic_extraction_model,
)

search_string = sesg.generate(
    [
        "machine,learning,artificial,intelligence",
        "computer,science,graph,theory",
    ],
)


print(search_string)
# '("machine" AND "learning") OR ("computer" AND "science")'
```

## Development

Create a virtual environment:

```sh
python -m venv .venv
```

Activate the virtual environment:

```sh
source .venv/bin/activate  # if using linux
```

Install the project in editable mode:

```sh
pip install -e .
```

### Testing

Install test dependencies:

```sh
pip install ".[dev-test]"
```

Run the test command from the provided script:

```sh
./scripts/test.sh
```

After running the tests, a coverage report will be available in `htmlcov/index.html`. You can run the following command to open the report using google chrome:

```
google-chrome htmlcov/index.html
```
