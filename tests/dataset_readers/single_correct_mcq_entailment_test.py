# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from lib.dataset_readers import SingleCorrectMcqEntailmentReader

class SingleCorrectMcqEntailmentReaderTest(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = SingleCorrectMcqEntailmentReader()
        instances = reader.read('tests/fixtures/datasets/single-correct-mcq-entailment-dataset-fixture.jsonl')
        ensure_list(instances)
        assert len(instances) == 2
