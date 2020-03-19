import os
import random
import yaml
import numpy as np
import sklearn.preprocessing


def get_dataset_fn(fn):
    if not os.path.exists('data'):
        os.mkdir('data')
    return os.path.join('data', '%s' % fn)


def get_dataset(definition):
    collection_scheme = definition["collection_scheme"]
    if "dataset" in collection_scheme:
        fn = collection_scheme["dataset"]  # todo
    else:
        fn = '%s_random_d%s_nb%s.npy' % (
            collection_scheme["metric_type"], collection_scheme["dim"], collection_scheme["data_size"])

    fn = get_dataset_fn(fn)
    print(fn)
    if not os.path.exists(fn):
        print('generating datset...')
        dimension = collection_scheme["dim"]
        xb = collection_scheme["data_size"]
        insert_vectors = [[random.random() for _ in range(dimension)]
                          for _ in range(xb)]
        D = sklearn.preprocessing.normalize(insert_vectors, axis=1, norm='l2')
        np.save(fn, D)
    else:
        D = np.load(fn)
    return D[:].tolist(), D[:].tolist()


def get_definition_from_yaml(fn):
    with open(fn, 'r') as f:
        definitions = yaml.load(f, yaml.SafeLoader)
    return definitions["QPS"]
