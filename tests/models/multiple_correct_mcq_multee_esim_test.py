# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class SingleCorrectMcqMulteeEsimTest(ModelTestCase):
    def setUp(self):
        super(SingleCorrectMcqMulteeEsimTest, self).setUp()
        self.set_up_model('tests/fixtures/experiment_configs/multiple_correct_mcq_multee.json',
                          'tests/fixtures/datasets/multiple-correct-mcq-entailment-dataset-fixture.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
