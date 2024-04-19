# my_dag.py

import os
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain.document_loaders import WikipediaLoader
from langchain.graphs import FalkorDBGraph
from langchain_openai import OpenAI
from langchain.chains import FalkorDBQAChain
from types import ModuleType
import pickle

os.environ['OPENAI_API_KEY']="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"

def config() -> dict:
    return {"query" : "Warren Buffett",
            "graph_type" : "custom"}

def diffbot_setup() -> ModuleType:
    diffbot_api_key = "a048854dd6c18c04c376beffea4165d5"
    diffbot_nlp = DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)
    return diffbot_nlp

def load_data(diffbot_setup: ModuleType, config: dict) -> list:
    if config['graph_type'] == "custom":
        return None
    query = config['query']
    # query = "Warren Buffett"
    # raw_documents = WikipediaLoader(query=query).load()
    # graph_documents = diff_bot_setup.convert_to_graph_documents(raw_documents)
    # with open('graph.pkl', 'wb') as f:
    #     pickle.dump(graph_documents, f)
    with open('graph.pkl', 'rb') as f:
        graph_documents = pickle.load(f)
    return graph_documents

def setup_falkordb(load_data: list, config: dict) -> ModuleType:
    if config['graph_type'] == "diffbot":
        graph = FalkorDBGraph("falkordb")
        graph.add_graph_documents(load_data)
        graph.refresh_schema()
        return graph
    else:
        graph = FalkorDBGraph("movies")
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
    # graph = setup_falkordb()
    chain = FalkorDBQAChain.from_llm(OpenAI(temperature=0), 
                                     graph=setup_falkordb, 
                                     verbose=True)
    return chain.run(get_input_query)