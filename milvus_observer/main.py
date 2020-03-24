# from __future__ import absolute_import
import os
import sys
import argparse
import pprint as pp

import pdb
import traceback

from milvus_observer.utils import positive_int
from milvus_observer.utils import get_definition_from_yaml
from milvus_observer.runner import run
from milvus_observer.log import Log

log = Log(__name__)
logger = log.Logger


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--suite',
        metavar='FILE',
        help='load test suite from FILE',
        required=True)
    parser.add_argument(
        '--queryfile',
        metavar='FILE',
        help='load query conf from FILE',
        required=False)
    parser.add_argument(
        '--collection',
        metavar='NAME',
        help='run only the named collection',
        default=None)
    parser.add_argument(
        "-testsize",
        type=positive_int,
        help="the query vector size",
        default=None)
    parser.add_argument(
        "-k", "--topk",
        type=positive_int,
        help="the number of near neighbours to search for",
        default=None)
    parser.add_argument(
        "--clients",
        default=1,
        type=positive_int,
        help="the number of client")
    parser.add_argument(
        '--batch',
        action='store_true',
        help='If set, algorithms get all queries at once')
    parser.add_argument(
        '--runs',
        metavar='COUNT',
        type=positive_int,
        help='run each algorithm instance %(metavar)s times and use only'
             ' the best result',
        default=2)
    parser.add_argument(
        '--searchonly',
        action='store_true',
        help='search specified collection'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='use local milvus server'
    )
    parser.add_argument(
        '--host',
        help='server host ip param for local mode',
        default='127.0.0.1')
    parser.add_argument(
        '--port',
        help='server port param for local mode',
        default='19530')

    args = parser.parse_args()
    logger.debug("Args: %s", args)

    definitions = get_definition_from_yaml(args.suite)
    logger.debug("Raw: %s", definitions)

    if args.queryfile:
        query_definitions = get_definition_from_yaml(args.queryfile)
        definitions = parse_definitions(definitions, query_definitions)

    if args.collection:
        definitions = [
            d for d in definitions if d["collection_scheme"]["collection_name"] == args.collection]
        #  print(definitions)

    for d in definitions:
        if args.topk:
            d["search_args"]["topk"] = [args.topk]
        if args.testsize:
            d["search_args"]["testsize"] = [args.testsize]
    logger.debug("Definitions: %s" % definitions)

    if args.queryfile:
        for definition in definitions:
            pp.pprint(definition)
            run(definition, definition["clients"],
                definition["runs"], definition["batch"], True)
    else:
        for definition in definitions:
            run(definition, args.clients, args.runs, args.batch, args.searchonly)


def parse_definitions(suite, querys):
    query_collection = [q["collection_name"] for q in querys["case"]]
    definitions = [
        d for d in suite if d["collection_scheme"]["collection_name"] in query_collection]
    for d in definitions:
        for q in querys["case"]:
            if d["collection_scheme"]["collection_name"] == q["collection_name"]:
                d["search_args"] = q["search_args"]
                d["clients"] = q["clients"] if "clients" in q else querys["clients"]
                d["runs"] = q["runs"] if "runs" in q else querys["runs"]
                d["batch"] = q["batch"] if "batch" in q else querys["batch"]
    return definitions
