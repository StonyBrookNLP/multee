import os
import sys
from typing import List, Callable

import boto3
from elasticsearch import Elasticsearch, RequestsHttpConnection

# A PremiseRetriever function takes a question stem and a question choice as input, and returns a list of
# premises.
from requests_aws4auth import AWS4Auth

PremiseRetriever = Callable[[str, str], List[str]]


# hard_coded returns a PremiseRetriever function that always returns the same set of premises.
#
# It is meant for testing and as an example that meets the expected interface. You can copy and changed it to
# do actual retrieval somehow.
def hard_coded() -> PremiseRetriever:
    premises = [
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

    def retrieve(question_stem: str, choice_text: str) -> List[str]:
        return premises

    return retrieve


# elasticsearch returns a PremiseRetriever function that retrieves premises by using an Elasticsearch client instance.
#
# * client is the Elasticsearch client to use.
# * index is the Elasticsearch index to query
# * document_type are the documents types that will be retrieved
# * field_name is the name of the field that is searched and retrieved
#
# Output: a function that matches the signature PremiseRetriever
def elasticsearch(client: Elasticsearch, index: str, document_type: str, field_name: str) -> PremiseRetriever:
    # Hard-coded retrieval configuration. Possibly parametrize this.
    max_hits = 10
    max_stem_length = 40

    def retrieve(stem: str, choice: str) -> List[str]:
        # Docs for syntax: https://www.elastic.co/guide/en/elasticsearch/reference/6.7/query-dsl.html
        r = client.search(
            index=index,
            body={
                "from": 0,
                "size": max_hits,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {field_name: stem[-max_stem_length:] + " " + choice}
                            }
                        ],
                        "filter": [
                            {"type": {"value": document_type}},
                            {"match": {field_name: stem}},
                            {"match": {field_name: choice}}
                        ],
                    }
                }
            }
        )

        hits = r["hits"]
        if hits["total"] == 0:
            return []

        sentences = []
        for hit in hits["hits"]:
            sentences.append(hit["_source"][field_name])

        return sentences

    return retrieve


# Returns a PremiseRetriever that queries an AWS Elasticsearch instance, configured with env variables
def aws_elasticsearch() -> PremiseRetriever:
    env_var_aws_es_hostname = "AWS_ES_HOSTNAME"
    env_var_aws_es_region = "AWS_ES_REGION"
    env_var_aws_es_index = "AWS_ES_INDEX"
    env_var_aws_es_document_type = "AWS_ES_DOCUMENT_TYPE"
    env_var_aws_es_field_name = "AWS_ES_FIELD_NAME"

    needed_env_vars = [
        env_var_aws_es_hostname,
        env_var_aws_es_region,
        env_var_aws_es_index,
        env_var_aws_es_document_type,
        env_var_aws_es_field_name
    ]
    for var in needed_env_vars:
        if var not in os.environ:
            raise Exception(f"Missing env var {var}")

    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, os.environ[env_var_aws_es_region], 'es')
    client = Elasticsearch(
        hosts=[{'host': os.environ[env_var_aws_es_hostname], 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    return elasticsearch(
        client=client,
        index=os.environ[env_var_aws_es_index],
        document_type=os.environ[env_var_aws_es_document_type],
        field_name=os.environ[env_var_aws_es_field_name]
    )


# Returns a PremiseRetriever by inspecting environment variables
def from_environment() -> PremiseRetriever:
    env_var_premise_retriever = "MULTEE_PREMISE_RETRIEVER"

    if env_var_premise_retriever not in os.environ:
        raise Exception(f"Missing env var {env_var_premise_retriever}")

    name = os.environ[env_var_premise_retriever]
    if name == "aws-es":
        return aws_elasticsearch()

    if name == "hard-coded":
        return hard_coded()

    raise Exception(f"Unexpected premise retriever in env var {env_var_premise_retriever}: {name}")
