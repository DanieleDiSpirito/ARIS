import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Configurazione
MODELS = ["BAAI/bge-m3"] # Il modello locale che hai scelto
DBS = ["db_nitro", "db_standard", "db_exacto"]
TEST_FILE = "test_questions.csv"

def calcola_hit_rate(db_name):
    # Connessione al DB specifico
    client = chromadb.PersistentClient(path=f"./vector_db/{db_name}")
    emb_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
    collection = client.get_collection(name="manuali_fanuc", embedding_function=emb_fn)
    
    # Caricamento domande
    df = pd.read_csv(TEST_FILE)
    hits = 0
    
    for _, row in df.iterrows():
        # Query al database
        results = collection.query(query_texts=[row['question']], n_results=3)
        
        # Verifica nei metadati
        trovato = False
        for meta in results['metadatas'][0]:
            if str(meta['page']) == str(row['expected_page']) and meta['file_name'] == row['expected_file']:
                trovato = True
                break
        
        if trovato:
            hits += 1
            
    return (hits / len(df)) * 100

# Esecuzione test
for db in DBS:
    score = calcola_hit_rate(db)
    print(f"Risultato {db}: Hit Rate@3 = {score}%")