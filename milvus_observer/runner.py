import datetime
import numpy
import os
import sys
import time
import concurrent.futures
import milvus
import numpy

from milvus_observer.client import MilvusClient
from milvus_observer.data import get_dataset
from milvus_observer.utils import generate_combinations

INSERT_INTERVAL = 50000


def run_individual_query(connect, query, search_param, batch):
    start = time.time()
    result = connect.query(
        query, search_param["topk"], search_param=search_param)
    total = (time.time() - start)
    attrs = {
        "totoal_time": total
    }
    return attrs
    # return (attrs, result)


def build_all(connect, X_train, collection_scheme, build_param):
    if connect.exists_collection():
        connect.delete()
        time.sleep(2)
    connect.create_collection(
        collection_scheme["collection_name"], collection_scheme["dim"],
        collection_scheme["index_size"], collection_scheme["metric_type"])
    loops = len(X_train) // INSERT_INTERVAL + 1
    for i in range(loops):
        start = i*INSERT_INTERVAL
        end = min((i+1)*INSERT_INTERVAL, len(X_train))
        tmp_vectors = X_train[start:end]
        if start < end:
            connect.insert(
                tmp_vectors, ids=[i for i in range(start, end)])
        connect.flush()
    if connect.count() != len(X_train):
        print("Table row count is not equal to insert vectors")
        return
    connect.create_index(
        collection_scheme["index_type"], index_param=build_param)
    connect.preload_collection()


def run(definition, connection_num, run_count, batch):
    collection_scheme = definition["collection_scheme"]

    pool = [MilvusClient(collection_name=collection_scheme["collection_name"])
            for n in range(connection_num)]
    X_train, X_test = get_dataset(definition)
    build_params = generate_combinations(definition["build_args"])
    search_params = generate_combinations(definition["search_args"])
    for pos, build_param in enumerate(build_params, 1):
        print("Running train argument group %d of %d..." %
              (pos, len(build_params)))
        print("build_params:", build_param)
        build_all(pool[0], X_train, collection_scheme, build_param)
        for pos, search_param in enumerate(search_params, 1):
            print("Running search argument group %d of %d..." %
                  (pos, len(search_params)))
            print("search_params:", search_param)
            query_vector = X_test[0:search_param["query_size"]]
            min_total_time = 100
            for _ in range(run_count):
                totoal_time = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=connection_num) as executor:
                    future_results = {executor.submit(
                        run_individual_query, pool[pos], query_vector, search_param, batch): pos for pos in range(connection_num)}
                    for future in concurrent.futures.as_completed(future_results):
                        data = future.result()
                        totoal_time = totoal_time if data["totoal_time"] < totoal_time else data["totoal_time"]
                    min_total_time = min_total_time if min_total_time < totoal_time else totoal_time
            average_search_time = min_total_time / \
                (connection_num * search_param["query_size"])
            print("QPS: %s\n" % (1.0 / average_search_time))