import os
import yaml
from itertools import product
import time


def generate_combinations(args):
    if isinstance(args, list):
        args = [el if isinstance(el, list) else [el] for el in args]
        return [list(x) for x in product(*args)]
    elif isinstance(args, dict):
        flat = []
        for k, v in args.items():
            if isinstance(v, list):
                flat.append([(k, el) for el in v])
            else:
                flat.append([(k, v)])
        return [dict(x) for x in product(*flat)]
    else:
        raise TypeError("No args handling exists for %s" %
                        type(args).__name__)


if __name__ == "__main__":
    # fn = "example_suite.yaml"
    # print("file: %s\n" % (1.0/0.2))
    # with open(fn, 'r') as f:
    #     definitions = yaml.load(f, yaml.SafeLoader)
    # for point in definitions:
    #     for metric in definitions[point]:
    #         comb = generate_combinations(metric["search_args"])
    #         for e in comb:
    #             print("search_params:", e, "\n")
    # t0 = time.time()
    # t1 = time.time()
    # print("time:%f" % (t1-t0))

    nq = 1
    connection_num = 1
    batch_size = int(nq / connection_num)
    print("batch_size:",batch_size)
    for pos in range(connection_num):
        start = pos * batch_size
        end = (pos+1) * batch_size
        print("pos %d: from %d to %d" % (pos, start, end))

