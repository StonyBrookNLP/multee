from typing import List, Callable
from elasticsearch import Elasticsearch

# A PremiseRetriever function takes a question stem and a question choice as input, and returns a list of
# premises.
PremiseRetriever = Callable[[str, str], List[str]]


# hard_coded returns a PremiseRetriever function that always returns the same set of premises.
#
# It is meant for testing and as an example that meets the expected interface. You can copy and changed it to
# do actual retrieval somehow.
def hard_coded(premises: List[str]) -> PremiseRetriever:
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
