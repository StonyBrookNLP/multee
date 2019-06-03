from typing import List
import logging
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Predictor.register("entailment_pair")
class EntailmentPairPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._entailment_idx = self._model.vocab.get_token_index("entailment", "labels")
        self._contradiction_idx = self._model.vocab.get_token_index("contradiction", "labels")
        self._neutral_idx = self._model.vocab.get_token_index("neutral", "labels")

    @overrides
    def _json_to_instance(self,  # type: ignore
                          json_dict: JsonDict) -> Instance:
        premise_text = json_dict.get("sentence1", None) or json_dict.get("premise", None)
        hypothesis_text = json_dict.get("sentence2", None) or json_dict.get("hypothesis", None)
        if premise_text and hypothesis_text:
            return self._dataset_reader.text_to_instance(premise_text, hypothesis_text)
        logger.info("Error parsing input")
        return None

    @overrides
    def predict_json(self, inputs: JsonDict):
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        inputs["entailment_prob"] = float(outputs["label_probs"][self._entailment_idx])
        inputs["contradiction_prob"] = float(outputs["label_probs"][self._contradiction_idx])
        inputs["neutral_prob"] = float(outputs["label_probs"][self._neutral_idx])
        return sanitize(inputs)

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self.predict_batch_instance(instances)
        for input, output in zip(inputs, outputs):
            input["entailment_prob"] = float(output["label_probs"][self._entailment_idx])
            input["contradiction_prob"] = float(output["label_probs"][self._contradiction_idx])
            input["neutral_prob"] = float(output["label_probs"][self._neutral_idx])
        return inputs

