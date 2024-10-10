from datetime import datetime, timezone
import pandas as pd
from elasticsearch import Elasticsearch
import numpy as np

def convert_epoch_to_isoformat(epoch_time):
    if not isinstance(epoch_time, (float, int)):
        raise TypeError("Epoch time must be a float or an integer.")
    return datetime.fromtimestamp(epoch_time, tz=timezone.utc).isoformat()

def clean_doc(doc):
    # Convert NaN values to None or appropriate default values
    for key, value in doc.items():
        if pd.isna(value):
            if key in ['post_score', 'post_upvote_ratio']:  # Assuming you want to set missing numeric values to 0
                doc[key] = 0
            else:
                doc[key] = None  # Elasticsearch will ignore fields with None values

    # Convert floating point to integer for post_score if necessary
    doc['post_score'] = int(doc['post_score']) if pd.notnull(doc['post_score']) else 0

    return doc

def main():
    es = Elasticsearch("http://elasticsearch:9200")


    df = pd.read_excel("/app/elasticsearch/data/labelled_full_corpus.xlsx")

    mapping = {
        "mappings": {
            "properties": {
                "subreddit": {"type": "keyword"},
                "post_id": {"type": "keyword"},
                "post_title": {"type": "text"},
                "post_body": {"type": "text"},
                "post_created": {"type": "date"},
                "post_score": {"type": "integer"},
                "post_upvote_ratio": {"type": "float"},
                "post_url": {"type": "text"},
                "subjectivity_label": {"type": "keyword"},
                "sentiment_label": {"type": "keyword"},
                "combined_label": {"type": "keyword"}
            }
        }
    }

    if not es.indices.exists(index="posts"):
        es.indices.create(index="posts", body=mapping)

    for index, row in df.iterrows():
        doc = row.to_dict()

        if pd.notnull(doc['post_created']):
            try:
                doc['post_created'] = convert_epoch_to_isoformat(doc['post_created'])
            except TypeError:
                print(f"Invalid timestamp for document {doc['post_id']}: {doc['post_created']}")
                continue

        doc = clean_doc(doc)

        try:
            es.index(index="posts", id=doc['post_id'], body=doc)
            # print(doc)
        except Exception as e:
            print(f"Error indexing document {doc['post_id']}: {e}")
            print(doc)
    try:
        es.indices.put_settings(index="posts", body={"index.max_result_window": df.shape[0]})
    except Exception as e:
        print("Could not update max result window")

if __name__ == "__main__":
    main()
