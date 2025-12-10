from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run("RETURN 1 AS num")
        record = result.single()
        if record and record["num"] == 1:
            print("Successfully connected to Neo4j!")
        else:
            print("Connected, but unexpected result.")
    driver.close()
except Exception as e:
    print(e)
