from typing import List
import logging
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Predictor.register("multiple_correct_mcq_entailment")
class MultipleCorrectMcqEntailment(Predictor):

    @overrides
    def _json_to_instance(self,  # type: ignore
                          json_dict: JsonDict) -> Instance:
        premises = json_dict["premises"]
        hypotheses = json_dict["hypotheses"]
        entailments = json_dict.get("entailments", None)
        if entailments is None:
            answer_indices = None
        else:
            answer_indices = [index for index, entailment in enumerate(entailments) if entailment]
        relevant_sentence_idxs = json_dict.get("relevant_sentence_idxs", None)
        return self._dataset_reader.text_to_instance(premises,
                                                     hypotheses,
                                                     answer_indices,
                                                     relevant_sentence_idxs)

    @overrides
    def predict_json(self, input: JsonDict):
        instance = self._json_to_instance(input)
        output = self._model.forward_on_instance(instance)
        return_json = {}
        return_json["input"] = input

        label_probs = output["label_probs"]
        predicted_answer_indices = [index for index, prob in enumerate(list(label_probs)) if prob >= 0.5]
        premises_attentions = output.get("premises_attentions", None)
        premises_aggregation_attentions = output.get("premises_aggregation_attentions", None)

        return_json["label_probs"] = label_probs
        return_json["predicted_answer_indices"] = predicted_answer_indices
        if premises_attentions is not None:
            return_json["premises_attentions"] = premises_attentions
            return_json["premises_aggregation_attentions"] = premises_aggregation_attentions
        return sanitize(return_json)

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self._model.forward_on_instances(instances)
        return_jsons = []
        for input, output in zip(inputs, outputs):
            return_json = {}
            return_json["input"] = input
            premises_count = len(input["premises"])

            label_probs = output["label_probs"]
            predicted_answer_indices = [index for index, prob in enumerate(list(label_probs)) if prob >= 0.5]
            premises_attentions = output.get("premises_attentions", None)
            premises_aggregation_attentions = output.get("premises_aggregation_attentions", None)

            return_json["label_probs"] = label_probs
            return_json["predicted_answer_indices"] = predicted_answer_indices
            if premises_attentions is not None:
                return_json["premises_attentions"] = premises_attentions[:, :premises_count]
                return_json["premises_aggregation_attentions"] = premises_aggregation_attentions[:, :premises_count]

            return_jsons.append(return_json)
        return sanitize(return_jsons)
