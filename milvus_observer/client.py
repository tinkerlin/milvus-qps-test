import sys
import random
import json
import time
import datetime
from milvus_observer.log import Log
from milvus import Milvus, IndexType, MetricType

import pdb

log = Log(__name__)
logger = log.Logger

SERVER_HOST_DEFAULT = "127.0.0.1"
SERVER_PORT_DEFAULT = 19530
INDEX_MAP = {
    "flat": IndexType.FLAT,
    "ivf_flat": IndexType.IVFLAT,
    "ivf_sq8": IndexType.IVF_SQ8,
    "nsg": IndexType.RNSG,
    "ivf_sq8h": IndexType.IVF_SQ8H,
    "ivf_pq": IndexType.IVF_PQ,
    "hnsw": IndexType.HNSW
}


def time_wrapper(func):
    """
    This decorator prints the execution time for the decorated function.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info("Milvus {} run in {}s".format(
            func.__name__, round(end - start, 2)))
        return result
    return wrapper


class MilvusClient(object):
    def __init__(self, collection_name=None, ip=None, port=None, timeout=60):
        self._milvus = Milvus()
        self._collection_name = collection_name
        try:
            i = 1
            start_time = time.time()
            if not ip:
                self._milvus.connect(
                    host=SERVER_HOST_DEFAULT,
                    port=SERVER_PORT_DEFAULT)
            else:
                # retry connect for remote server
                while time.time() < start_time + timeout:
                    try:
                        self._milvus.connect(
                            host=ip,
                            port=port)
                        if self._milvus.connected() is True:
                            logger.debug("Try connect times: %d, %s" % (
                                i, round(time.time() - start_time, 2)))
                            break
                    except Exception as e:
                        logger.debug("Milvus connect failed")
                        i = i + 1

        except Exception as e:
            raise e

    def __str__(self):
        return 'Milvus collection %s' % self._collection_name

    def check_status(self, status):
        if not status.OK():
            logger.error(status.message)
            raise Exception("Status not ok")

    def create_collection(self, collection_name, dimension, index_file_size, metric_type):
        if not self._collection_name:
            self._collection_name = collection_name
        if metric_type == "l2":
            metric_type = MetricType.L2
        elif metric_type == "ip":
            metric_type = MetricType.IP
        elif metric_type == "jaccard":
            metric_type = MetricType.JACCARD
        elif metric_type == "hamming":
            metric_type = MetricType.HAMMING
        else:
            logger.error("Not supported metric_type: %s" % metric_type)
        create_param = {'collection_name': collection_name,
                        'dimension': dimension,
                        'index_file_size': index_file_size,
                        "metric_type": metric_type}
        status = self._milvus.create_collection(create_param)
        self.check_status(status)

    @time_wrapper
    def insert(self, X, ids=None):
        status, result = self._milvus.add_vectors(
            self._collection_name, X, ids)
        self.check_status(status)
        return status, result

    @time_wrapper
    def delete_vectors(self, ids):
        status = self._milvus.delete_by_id(self._collection_name, ids)
        self.check_status(status)

    @time_wrapper
    def flush(self):
        status = self._milvus.flush([self._collection_name])
        self.check_status(status)

    @time_wrapper
    def compact(self):
        status = self._milvus.compact(self._collection_name)
        self.check_status(status)

    @time_wrapper
    def create_index(self, index_type, index_param=None):
        index_type = INDEX_MAP[index_type]
        logger.info("Building index start, collection_name: %s, index_type: %s" % (
            self._collection_name, index_type))
        if index_param:
            logger.info(index_param)
        status = self._milvus.create_index(
            self._collection_name, index_type, index_param)
        self.check_status(status)

    def describe_index(self):
        status, result = self._milvus.describe_index(self._collection_name)
        self.check_status(status)
        index_type = None
        for k, v in INDEX_MAP.items():
            if result._index_type == v:
                index_type = k
                break
        return {"index_type": index_type, "index_param": result._params}

    def drop_index(self):
        logger.info("Drop index: %s" % self._collection_name)
        return self._milvus.drop_index(self._collection_name)

    @time_wrapper
    def query(self, X, top_k, search_param=None):
        status, result = self._milvus.search_vectors(
            self._collection_name, top_k, query_records=X, params=search_param)
        self.check_status(status)
        return result

    def count(self):
        return self._milvus.count_collection(self._collection_name)[1]

    def delete(self, timeout=120):
        timeout = int(timeout)
        logger.info("Start delete collection: %s" % self._collection_name)
        self._milvus.drop_collection(self._collection_name)
        i = 0
        while i < timeout:
            if self.count():
                time.sleep(1)
                i = i + 1
                continue
            else:
                break
        if i >= timeout:
            logger.error("Delete collection timeout")

    def describe(self):
        return self._milvus.describe_collection(self._collection_name)

    def show_collections(self):
        return self._milvus.show_collections()

    def exists_collection(self, collection_name=None):
        if collection_name is None:
            collection_name = self._collection_name
        status, res = self._milvus.has_collection(collection_name)
        # self.check_status(status)
        return res

    @time_wrapper
    def preload_collection(self):
        return self._milvus.preload_collection(self._collection_name, timeout=3000)

    def get_server_version(self):
        status, res = self._milvus.server_version()
        return res

    def get_server_mode(self):
        return self.cmd("mode")

    def get_server_commit(self):
        return self.cmd("build_commit_id")

    def get_server_config(self):
        return json.loads(self.cmd("get_config *"))

    def get_mem_info(self):
        result = json.loads(self.cmd("get_system_info"))
        result_human = {
            # unit: Gb
            "memory_used": round(int(result["memory_used"]) / (1024*1024*1024), 2)
        }
        return result_human

    def cmd(self, command):
        status, res = self._milvus._cmd(command)
        logger.info("Server command: %s, result: %s" % (command, res))
        self.check_status(status)
        return res
