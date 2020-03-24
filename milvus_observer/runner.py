import datetime
import numpy
import os
import sys
import time
import concurrent.futures
import milvus
import numpy

from milvus_observer.client import MilvusClient
from milvus_observer.dataset import get_dataset
from milvus_observer.utils import generate_combinations

INSERT_INTERVAL = 50000


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


def run(definition, connection_num, run_count, batch, searchonly):
    collection_scheme = definition["collection_scheme"]

    X_train, X_test = get_dataset(definition)
    build_params = generate_combinations(definition["build_args"])
    search_params = generate_combinations(definition["search_args"])
    if not searchonly:
        for pos, build_param in enumerate(build_params, 1):
            print("Running train argument group %d of %d..." %
                  (pos, len(build_params)))
            print("build_params:", build_param)
            client = MilvusClient(
                collection_name=collection_scheme["collection_name"])
            build_all(client, X_train, collection_scheme, build_param)
            run_paralle(
                search_params, collection_scheme["collection_name"], connection_num, X_test, run_count, batch)
    else:
        run_paralle(
            search_params, collection_scheme["collection_name"], connection_num, X_test, run_count, batch)


# def run_paralle(search_params, collection_name, connection_num, X_test, run_count, batch):
#     pool = [MilvusClient(collection_name=collection_name)
#             for n in range(connection_num)]
#     for pos, search_param in enumerate(search_params, 1):
#         print("Running search argument group %d of %d..." %
#               (pos, len(search_params)))
#         print("search_params:", search_param)
#         if search_param["query_size"] == 1:
#             query_vector = [X_test[0]]
#         else:
#             query_vector = X_test[0:search_param["query_size"]]
#         min_total_time = float('inf')
#         for _ in range(run_count):
#             total_time = float('-inf')
#             with concurrent.futures.ThreadPoolExecutor(max_workers=connection_num) as executor:
#                 future_results = {executor.submit(
#                     run_individual_query, pool[pos], query_vector, search_param, batch): pos for pos in range(connection_num)}
#                 for future in concurrent.futures.as_completed(future_results):
#                     data = future.result()
#                     total_time = total_time if total_time > data["total_time"] else data["total_time"]
#                 min_total_time = min_total_time if min_total_time < total_time else total_time
#         average_search_time = min_total_time / \
#             (connection_num * search_param["query_size"])
#         print("QPS: %d\n" % (1.0 / average_search_time))


def run_paralle(search_params, collection_name, connection_num, X_test, run_count, batch):
    pool = [MilvusClient(collection_name=collection_name)
            for n in range(connection_num)]
    for pos, search_param in enumerate(search_params, 1):
        print("Running search argument group %d of %d..." %
              (pos, len(search_params)))
        print("search_params:", search_param)
        if search_param["query_size"] == 1:
            query_vector = [X_test[0]]
        else:
            query_vector = X_test[0:search_param["query_size"]]

        batch_size = int(search_param["query_size"]/connection_num)
        if batch_size <= 0:
            print("Error: test_size < clients num")
            exit()

        min_total_time = float('inf')
        for _ in range(run_count):
            total_time = float('-inf') 
            with concurrent.futures.ThreadPoolExecutor(max_workers=connection_num) as executor:
                future_results = {executor.submit(
                    run_individual_query, pool[pos], query_vector[(pos * batch_size):(pos*batch_size + batch_size)], search_param, batch): pos for pos in range(connection_num)}
                for future in concurrent.futures.as_completed(future_results):
                    data = future.result()
                    total_time = total_time if total_time > data["total_time"] else data["total_time"]
                min_total_time = min_total_time if min_total_time < total_time else total_time
        average_search_time = min_total_time / (batch_size * connection_num)
        print("QPS: %d\n" % (1.0 / average_search_time))


def run_individual_query(connect, query, search_param, batch):
    start = time.time()
    if batch:
        connect.query(
            query, search_param["topk"], search_param=search_param)
    else:
        [connect.query([x], search_param["topk"],
                       search_param=search_param) for x in query]
    total = (time.time() - start)
    attrs = {
        "total_time": total
    }
    return attrs
