# This is a fork of the AllenNLP simple server program
# https://github.com/allenai/allennlp/blob/master/allennlp/service/server_simple.py
#
# For invocation, see start-server*.sh in the parent directory.

import argparse
import json
import logging
import os
import re
import sys

from flask import Flask, request, Response, jsonify, send_file, send_from_directory
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from allennlp.common import JsonDict
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from hypothesis.explicit import explicit_hypothesis
from hypothesis.implicit import implicit_hypothesis
from premise_retriever import retrievers

from typing import Callable
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ServerError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict['message'] = self.message
        return error_dict


def make_app(predictor: Predictor,
             premise_retriever: retrievers.PremiseRetriever,
             static_dir: str = None,
             sanitizer: Callable[[JsonDict], JsonDict] = None) -> Flask:
    """
    Creates a Flask app that serves up the provided ``Predictor``
    along with a front-end for interacting with it.
    If you want to use the built-in bare-bones HTML, you must provide the
    field names for the inputs (which will be used both as labels
    and as the keys in the JSON that gets sent to the predictor).
    If you would rather create your own HTML, call it index.html
    and provide its directory as ``static_dir``. In that case you
    don't need to supply the field names -- that information should
    be implicit in your demo site. (Probably the easiest thing to do
    is just start with the bare-bones HTML and modify it.)
    In addition, if you want somehow transform the JSON prediction
    (e.g. by removing probabilities or logits)
    you can do that by passing in a ``sanitizer`` function.
    """
    static_dir = os.path.abspath(static_dir)
    if not os.path.exists(static_dir):
        logger.error("app directory %s does not exist, aborting", static_dir)
        sys.exit(-1)

    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response:  # pylint: disable=unused-variable
        return send_file(os.path.join(static_dir, 'index.html'))

    @app.route('/predict-internal', methods=['POST', 'OPTIONS'])
    def predict_internal() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        internal_request = request.get_json()

        prediction = predictor.predict_json(internal_request)
        if sanitizer is not None:
            prediction = sanitizer(prediction)

        log_blob = {"inputs": internal_request, "outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))

        return jsonify(prediction)

    def split_mc_question(question, min_choices=2):
        choice_sets = [["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"],
                       ["1", "2", "3", "4", "5"],
                       ["G", "H", "J", "K"],
                       ['a', 'b', 'c', 'd', 'e']]
        patterns = [r'\(#\)', r'#\)', r'#\.']
        for pattern in patterns:
            for choice_set in choice_sets:
                regex = pattern.replace("#", "([" + "".join(choice_set) + "])")
                labels = [m.group(1) for m in re.finditer(regex, question)]
                if len(labels) >= min_choices and labels == choice_set[:len(labels)]:
                    splits = [s.strip() for s in re.split(regex, question)]
                    return {"stem": splits[0],
                            "choices": [{"text": splits[i + 1],
                                         "label": splits[i]} for i in range(1, len(splits) - 1, 2)]}
        return None

    @app.route('/predict-mcq', methods=['POST', 'OPTIONS'])
    def predict_mcq() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()
        if not data["mcq"]:
            return Response(status=400, response="Missing mcq")

        if isinstance(data["mcq"], str):
            q = split_mc_question(data["mcq"])
        elif isinstance(data["mcq"], dict):
            q = data["mcq"]
        else:
            return Response(status=400, response="Unexpected type of mcq: " + str(type(data["mcq"])))

        internal_request = prepare_internal_request(q)

        prediction = predictor.predict_json(internal_request)
        if sanitizer is not None:
            prediction = sanitizer(prediction)

        log_blob = {"inputs": internal_request, "outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))

        return jsonify(prediction)

    def prepare_internal_request(q):
        ihypothesis = implicit_hypothesis(q["stem"])
        hypotheses = []
        for choice in q["choices"]:
            hypotheses.append(
                explicit_hypothesis(q["stem"], choice["text"], ihypothesis, False)
            )

        premises = []
        for c in q["choices"]:
            premises += premise_retriever(q["stem"], c["text"])

        # keep only unique premises, and sort them for consistency
        premises = sorted(set(premises))

        return {
            'premises': premises,
            'hypotheses': hypotheses,
        }

    @app.route('/<path:path>')
    def static_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        return send_from_directory(static_dir, path)

    return app


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_path,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)

    return Predictor.from_archive(archive, args.predictor)


ENV_VAR_AWS_ES_HOSTNAME = "AWS_ES_HOSTNAME"
ENV_VAR_AWS_ES_REGION = "AWS_ES_REGION"
ENV_VAR_AWS_ES_INDEX = "AWS_ES_INDEX"
ENV_VAR_AWS_ES_DOCUMENT_TYPE = "AWS_ES_DOCUMENT_TYPE"
ENV_VAR_AWS_ES_FIELD_NAME = "AWS_ES_FIELD_NAME"


def _get_premise_retriever(name) -> retrievers.PremiseRetriever:
    if name == "aws-es":
        needed_env_vars = [
            ENV_VAR_AWS_ES_HOSTNAME,
            ENV_VAR_AWS_ES_REGION,
            ENV_VAR_AWS_ES_INDEX,
            ENV_VAR_AWS_ES_DOCUMENT_TYPE,
            ENV_VAR_AWS_ES_FIELD_NAME
        ]
        for var in needed_env_vars:
            if var not in os.environ:
                print(f"Missing env var {var}")
                sys.exit(1)

        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, os.environ[ENV_VAR_AWS_ES_REGION], 'es')
        client = Elasticsearch(
            hosts=[{'host': os.environ[ENV_VAR_AWS_ES_HOSTNAME], 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

        return retrievers.elasticsearch(
            client=client,
            index=os.environ[ENV_VAR_AWS_ES_INDEX],
            document_type=os.environ[ENV_VAR_AWS_ES_DOCUMENT_TYPE],
            field_name=os.environ[ENV_VAR_AWS_ES_FIELD_NAME]
        )

    if name == "hard-coded":
        return retrievers.hard_coded(
            premises=[
                "Sky color is simply the color of the sky.",
                "It's the color of the sea, color of the sky.",
                "Put the sky reflection with the sky colors.",
                "The sky color, as the name implies, is the color of the sky in the environment.",
                "Thus having the color of the sky the color of the sky.",
                "Mix the color of the sky with the color of the Sun.",
                "The color is a medium sky blue color.",
                "Colors of the rainbow or the sky rising.",
                "A panoramic sunset colors the sky.",
                "Blue is the color of the sky.",
            ]
        )

    print(f"Unknown premise retriever: {name}")
    sys.exit(1)


def main(args):
    parser = argparse.ArgumentParser(description='Multee server')

    parser.add_argument('--archive-path', type=str, required=True, help='path to trained archive file')
    parser.add_argument('--predictor', type=str, required=True, help='name of predictor')
    parser.add_argument('--weights-file', type=str,
                        help='a path that overrides which weights file to use')
    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides', type=str, default="",
                        help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--static-dir', type=str, help='serve index.html from this directory')
    parser.add_argument('--port', type=int, default=8000, help='port to serve the demo on')
    parser.add_argument('--premise-retriever', type=str, default="hard-coded",
                        help='the kind of premise retriever to use')

    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')

    args = parser.parse_args(args)

    print(f"Preparing premise retriever...")
    premise_retriever: retrievers.PremiseRetriever = _get_premise_retriever(args.premise_retriever)

    print(f"Loading modules...")
    for package_name in args.include_package:
        import_submodules(package_name)

    print(f"Loading model...")
    predictor = _get_predictor(args)

    print(f"Starting server...")
    app = make_app(
        predictor=predictor,
        premise_retriever=premise_retriever,
        static_dir=args.static_dir
    )
    CORS(app)
    http_server = WSGIServer(('0.0.0.0', args.port), app)

    print(f"Serving demo on port {args.port}")
    http_server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
