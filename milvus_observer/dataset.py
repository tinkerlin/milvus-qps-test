import h5py
import numpy as np
import os
import random
import sys
import sklearn.preprocessing

from urllib.request import urlopen
from urllib.request import urlretrieve

from milvus_observer.distance import dataset_transform


def reporthook(blocknum, blocksize, totalsize):
    percent = 100.0 * blocknum * blocksize / totalsize
    if percent > 100:
        percent = 100
    print("%.2f%%" % percent)


def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst, reporthook=reporthook)


def get_dataset_fn(fn):
    if not os.path.exists('data'):
        os.mkdir('data')
    return os.path.join('data', '%s' % fn)


def get_dataset(definition):
    collection_scheme = definition["collection_scheme"]
    if "dataset" in collection_scheme:
        fn = collection_scheme["dataset"]
        return get_standard_dataset(fn)
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


def get_standard_dataset(fn):
    hdf5_fn = get_dataset_fn((fn + '.hdf5'))
    if not os.path.exists(fn):
        if fn in DATASETS:
            print("Creating dataset locally")
            DATASETS[fn](hdf5_fn)
        else:
            raise Exception("The dataset not in support list")
    D = h5py.File(hdf5_fn, 'r')
    X_train = np.array(D['train'])
    X_test = np.array(D['test'])
    distance = D.attrs['distance']
    X_train = dataset_transform[distance](X_train)
    X_test = dataset_transform[distance](X_test)
    return X_train[:].tolist(), X_test[:].tolist()


def write_output(train, test, fn, distance, point_type='float'):
    f = h5py.File(fn, 'w')
    f.attrs['distance'] = distance
    f.attrs['point_type'] = point_type
    print('train size: %9d * %4d' % train.shape)
    print('test size:  %9d * %4d' % test.shape)
    f.create_dataset('train', (len(train), len(
        train[0])), dtype=train.dtype)[:] = train
    f.create_dataset('test', (len(test), len(
        test[0])), dtype=test.dtype)[:] = test
    f.close()


def train_test_split(X, test_size=10000):
    import sklearn.model_selection
    print('Splitting %d*%d into train/test' % X.shape)
    return sklearn.model_selection.train_test_split(
        X, test_size=test_size, random_state=1)


def glove(out_fn, d):
    import zipfile

    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    fn = os.path.join('data', 'glove.twitter.27B.zip')
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(np.array(v))
        X_train, X_test = train_test_split(X)
        write_output(np.array(X_train), np.array(
            X_test), out_fn, 'angular')


def _load_texmex_vectors(f, n, k):
    import struct

    v = np.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack('f' * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t, fn):
    import struct
    m = t.getmember(fn)
    f = t.extractfile(m)
    k, = struct.unpack('i', f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
        test = _get_irisa_matrix(t, 'sift/sift_query.fvecs')
        write_output(train, test, out_fn, 'euclidean')


def gist(out_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz'
    fn = os.path.join('data', 'gist.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'gist/gist_base.fvecs')
        test = _get_irisa_matrix(t, 'gist/gist_query.fvecs')
        write_output(train, test, out_fn, 'euclidean')


def _load_mnist_vectors(fn):
    import gzip
    import struct

    print('parsing vectors in %s...' % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d")
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0]
                  for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = np.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0]
                        for j in range(entry_size)])
    return np.array(vectors)


def mnist(out_fn):
    download(
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist-train.gz')  # noqa
    download(
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'mnist-test.gz')  # noqa
    train = _load_mnist_vectors('mnist-train.gz')
    test = _load_mnist_vectors('mnist-test.gz')
    write_output(train, test, out_fn, 'euclidean')


def fashion_mnist(out_fn):
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',  # noqa
             'fashion-mnist-train.gz')
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',  # noqa
             'fashion-mnist-test.gz')
    train = _load_mnist_vectors('fashion-mnist-train.gz')
    test = _load_mnist_vectors('fashion-mnist-test.gz')
    write_output(train, test, out_fn, 'euclidean')

# Creates a 'deep image descriptor' dataset using the 'deep10M.fvecs' sample
# from http://sites.skoltech.ru/compvision/noimi/. The download logic is adapted
# from the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py.


def deep_image(out_fn):
    yadisk_key = 'https://yadi.sk/d/11eDCm7Dsn9GA'
    response = urlopen('https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key='
                       + yadisk_key + '&path=/deep10M.fvecs')
    response_body = response.read().decode("utf-8")

    dataset_url = response_body.split(',')[0][9:-1]
    filename = os.path.join('data', 'deep-image.fvecs')
    download(dataset_url, filename)

    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = np.fromfile(filename, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]

    X_train, X_test = train_test_split(fv)
    write_output(X_train, X_test, out_fn, 'angular')


def transform_bag_of_words(filename, n_dimensions, out_fn):
    import gzip
    from scipy.sparse import lil_matrix
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn import random_projection
    with gzip.open(filename, 'rb') as f:
        file_content = f.readlines()
        entries = int(file_content[0])
        words = int(file_content[1])
        file_content = file_content[3:]  # strip first three entries
        print("building matrix...")
        A = lil_matrix((entries, words))
        for e in file_content:
            doc, word, cnt = [int(v) for v in e.strip().split()]
            A[doc - 1, word - 1] = cnt
        print("normalizing matrix entries with tfidf...")
        B = TfidfTransformer().fit_transform(A)
        print("reducing dimensionality...")
        C = random_projection.GaussianRandomProjection(
            n_components=n_dimensions).fit_transform(B)
        X_train, X_test = train_test_split(C)
        write_output(np.array(X_train), np.array(
            X_test), out_fn, 'angular')


def nytimes(out_fn, n_dimensions):
    fn = 'nytimes_%s.txt.gz' % n_dimensions
    download('https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz', fn)  # noqa
    transform_bag_of_words(fn, n_dimensions, out_fn)


def random_float(out_fn, n_dims, n_samples, centers, distance):
    import sklearn.datasets

    X, _ = sklearn.datasets.make_blobs(
        n_samples=n_samples, n_features=n_dims,
        centers=centers, random_state=1)
    X_train, X_test = train_test_split(X, test_size=0.1)
    write_output(X_train, X_test, out_fn, distance)


def random_bitstring(out_fn, n_dims, n_samples, n_queries):
    import sklearn.datasets

    Y, _ = sklearn.datasets.make_blobs(
        n_samples=n_samples, n_features=n_dims,
        centers=n_queries, random_state=1)
    X = np.zeros((n_samples, n_dims), dtype=np.bool)
    for i, vec in enumerate(Y):
        X[i] = np.array([v > 0 for v in vec], dtype=np.bool)

    X_train, X_test = train_test_split(X, test_size=n_queries)
    write_output(X_train, X_test, out_fn, 'hamming', 'bit')


def word2bits(out_fn, path, fn):
    import tarfile
    local_fn = fn + '.tar.gz'
    url = 'http://web.stanford.edu/~maxlam/word_vectors/compressed/%s/%s.tar.gz' % (  # noqa
        path, fn)
    download(url, local_fn)
    print('parsing vectors in %s...' % local_fn)
    with tarfile.open(local_fn, 'r:gz') as t:
        f = t.extractfile(fn)
        n_words, k = [int(z) for z in next(f).strip().split()]
        X = np.zeros((n_words, k), dtype=np.bool)
        for i in range(n_words):
            X[i] = np.array([float(z) > 0 for z in next(
                f).strip().split()[1:]], dtype=np.bool)

        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, 'hamming', 'bit')


def sift_hamming(out_fn, fn):
    import tarfile
    local_fn = fn + '.tar.gz'
    url = 'http://sss.projects.itu.dk/ann-benchmarks/datasets/%s.tar.gz' % fn
    download(url, local_fn)
    print('parsing vectors in %s...' % local_fn)
    with tarfile.open(local_fn, 'r:gz') as t:
        f = t.extractfile(fn)
        lines = f.readlines()
        X = np.zeros((len(lines), 256), dtype=np.bool)
        for i, line in enumerate(lines):
            X[i] = np.array(
                [int(x) > 0 for x in line.decode().strip()], dtype=np.bool)
        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, 'hamming', 'bit')


def kosarak(out_fn):
    import gzip
    local_fn = 'kosarak.dat.gz'
    # only consider sets with at least min_elements many elements
    min_elements = 20
    url = 'http://fimi.uantwerpen.be/data/%s' % local_fn
    download(url, local_fn)

    with gzip.open('kosarak.dat.gz', 'r') as f:
        content = f.readlines()
        # preprocess data to find sets with more than 20 elements
        # keep track of used ids for reenumeration
        ids = {}
        next_id = 0
        cnt = 0
        for line in content:
            if len(line.split()) >= min_elements:
                cnt += 1
                for x in line.split():
                    if int(x) not in ids:
                        ids[int(x)] = next_id
                        next_id += 1

    X = np.zeros((cnt, len(ids)), dtype=np.bool)
    i = 0
    for line in content:
        if len(line.split()) >= min_elements:
            for x in line.split():
                X[i][ids[int(x)]] = 1
            i += 1

    X_train, X_test = train_test_split(np.array(X), test_size=500)
    write_output(X_train, X_test, out_fn, 'jaccard', 'bit')


def random_jaccard(out_fn, n=10000, size=50, universe=80):
    random.seed(1)
    l = list(range(universe))
    X = np.zeros((n, universe), dtype=np.bool)
    for i in range(len(X)):
        for j in random.sample(l, size):
            X[i][j] = True
    X_train, X_test = train_test_split(X, test_size=100)
    write_output(X_train, X_test, out_fn, 'jaccard', 'bit')


DATASETS = {
    'deep-image-96-angular': deep_image,
    'fashion-mnist-784-euclidean': fashion_mnist,
    'gist-960-euclidean': gist,
    'glove-25-angular': lambda out_fn: glove(out_fn, 25),
    'glove-50-angular': lambda out_fn: glove(out_fn, 50),
    'glove-100-angular': lambda out_fn: glove(out_fn, 100),
    'glove-200-angular': lambda out_fn: glove(out_fn, 200),
    'mnist-784-euclidean': mnist,
    'random-xs-20-euclidean': lambda out_fn: random_float(out_fn, 20, 10000, 100,
                                                          'euclidean'),
    'random-s-100-euclidean': lambda out_fn: random_float(out_fn, 100, 100000, 1000,
                                                          'euclidean'),
    'random-xs-20-angular': lambda out_fn: random_float(out_fn, 20, 10000, 100,
                                                        'angular'),
    'random-s-100-angular': lambda out_fn: random_float(out_fn, 100, 100000, 1000,
                                                        'angular'),
    'random-xs-16-hamming': lambda out_fn: random_bitstring(out_fn, 16, 10000,
                                                            100),
    'random-s-128-hamming': lambda out_fn: random_bitstring(out_fn, 128,
                                                            50000, 1000),
    'random-l-256-hamming': lambda out_fn: random_bitstring(out_fn, 256,
                                                            100000, 1000),
    'random-s-jaccard': lambda out_fn: random_jaccard(out_fn, n=10000,
                                                      size=20, universe=40),
    'random-l-jaccard': lambda out_fn: random_jaccard(out_fn, n=100000,
                                                      size=70, universe=100),
    'sift-128-euclidean': sift,
    'nytimes-256-angular': lambda out_fn: nytimes(out_fn, 256),
    'nytimes-16-angular': lambda out_fn: nytimes(out_fn, 16),
    'word2bits-800-hamming': lambda out_fn: word2bits(
        out_fn, '400K',
        'w2b_bitlevel1_size800_vocab400K'),
    'sift-256-hamming': lambda out_fn: sift_hamming(
        out_fn, 'sift.hamming.256'),
    'kosarak-jaccard': lambda out_fn: kosarak(out_fn),
}
