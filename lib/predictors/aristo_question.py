import logging
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance

from hypothesis.explicit import explicit_hypothesis
from hypothesis.implicit import implicit_hypothesis
from premise_retriever import retrievers

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

retriever = retrievers.from_environment()


@Predictor.register("aristo_question")
class AristoQuestion(Predictor):

    @overrides
    def predict_json(self, input: JsonDict) -> JsonDict:
        # The input should look like an Aristo question without an "answerKey" field:
        #
        #    {
        #        "id": "abc123",
        #        "question": {
        #            "stem": "What color is the sky?",
        #            "choices": [
        #                {"label": "A", "text": "red"},
        #                {"label": "B", "text": "green"},
        #                {"label": "C", "text": "blue"}
        #            ]
        #        }
        #    }
        #
        # And the returned value should look like an Aristo question prediction:
        #
        #    {
        #        "id": "abc123",
        #        "prediction": {
        #            "choices": [
        #                { "label": "A", "score": 0.6 },
        #                { "label": "B", "score": 0.3 },
        #                { "label": "C", "score": 0.7 }
        #            ]
        #        }
        #    }
        #
        # See https://json-schema.allenai.org/ for formal schema.

        premises, hypotheses = self.question_to_premises_and_hypotheses(input["question"])

        instance = self._json_to_instance({
            "premises": premises,
            "hypotheses": hypotheses
        })
        model_output = self._model.forward_on_instance(instance)

        pred_choices = []
        for choice, probability in zip(input["question"]["choices"], model_output["label_probs"]):
            pred_choices.append({"label": choice["label"], "score": probability})

        return sanitize({
            "id": input["id"],
            "prediction": {
                "choices": pred_choices
            },
            # Since other fields are permitted in the output, "diagnostics" is used here for a
            # representation of Multee's internal model output.
            "diagnostics": {
                "input": {
                    "premises": premises,
                    "hypotheses": hypotheses
                },
                "label_probs": model_output["label_probs"],
            }
        })

    @staticmethod
    def question_to_premises_and_hypotheses(question):
        stem = question["stem"]
        choice_texts = [c["text"] for c in question["choices"]]

        ihypothesis = implicit_hypothesis(stem)
        hypotheses = []
        for choice_text in choice_texts:
            hypotheses.append(
                explicit_hypothesis(stem, choice_text, ihypothesis, False)
            )

        premises = []
        for choice_text in choice_texts:
            premises += retriever(stem, choice_text)

        # keep only unique premises, and sort them for consistency
        premises = sorted(set(premises))

        return premises, hypotheses

    @overrides
    def _json_to_instance(self,  # type: ignore
                          json_dict: JsonDict) -> Instance:
        premises = json_dict["premises"]
        hypotheses = json_dict["hypotheses"]

        return self._dataset_reader.text_to_instance(premises, hypotheses, None, None)
