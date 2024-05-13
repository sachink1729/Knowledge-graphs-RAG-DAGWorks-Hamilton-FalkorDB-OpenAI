# RAG.py

import os
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain.document_loaders import WikipediaLoader
from langchain.graphs import FalkorDBGraph
from langchain_openai import OpenAI
from langchain.chains import FalkorDBQAChain
from types import ModuleType
import pickle

os.environ['OPENAI_API_KEY']="your api key"

def config() -> dict:
    return {"query" : "Warren Buffett",
            "graph_type" : "custom"}

def setup_falkordb() -> ModuleType:
    graph = FalkorDBGraph("movies")
# Populating knowledge graph
    graph.query(
    """
    CREATE
        (al:Person {name: 'Al Pacino', birthDate: '1940-04-25'}),
        (robert:Person {name: 'Robert De Niro', birthDate: '1943-08-17'}),
        (tom:Person {name: 'Tom Cruise', birthDate: '1962-07-3'}),
        (val:Person {name: 'Val Kilmer', birthDate: '1959-12-31'}),
        (anthony:Person {name: 'Anthony Edwards', birthDate: '1962-7-19'}),
        (meg:Person {name: 'Meg Ryan', birthDate: '1961-11-19'}),


        (god1:Movie {title: 'The Godfather'}),
        (god2:Movie {title: 'The Godfather: Part II'}),
        (god3:Movie {title: 'The Godfather Coda: The Death of Michael Corleone'}),
        (top:Movie {title: 'Top Gun'}),


        (al)-[:ACTED_IN]->(god1),
        (al)-[:ACTED_IN]->(god2),
        (al)-[:ACTED_IN]->(god3),
        (robert)-[:ACTED_IN]->(god2),
        (tom)-[:ACTED_IN]->(top),
        (val)-[:ACTED_IN]->(top),
        (anthony)-[:ACTED_IN]->(top),
        (meg)-[:ACTED_IN]->(top)
    """)
    return graph


def get_input_query() -> str:
    text = input("Write your query here->")
    return text


def get_response_from_openai(setup_falkordb: ModuleType, get_input_query: str) -> str:
    chain = FalkorDBQAChain.from_llm(OpenAI(temperature=0),
                                     graph=setup_falkordb,
                                     verbose=True)
    return chain.run(get_input_query)