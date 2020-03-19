Benchmarking Milvus QPS Performance
==============================
This project contains some tools to benchmark various implementations of ANNS algorithm already supported in milvus

Install
=======

The only prerequisite is Python and Milvus (tested with 3.6.8).

1. Clone the repo.
2. Run `pip install -r requirements.txt` .

Running
=======

1. Run `python run.py` (this can take an extremely long time)

You can customize the algorithms and datasets if you want to:

* Check that `example_suite.yaml` contains the parameter settings that you want to test
* To run specified suite, invoke `python run.py --suite=suites/your_own_suite.yaml ` . See `python run.py --help` for more information on possible settings. Note that experiments can take a long time.

Example
=======

### Please read example_suite.yaml first

1. Run `python run.py --suite=suites/example_suite.yaml` 
    - The script will build collection named **ivf_random_l2** and create dataset as descried in collection_scheme. The dataset is store at data folder. After insert data and build index, the scrpit will perform sereral parameter combinations to test QPS.
2. Once you run the suite.yaml, you don't need to recreate collection again. By invoke `python run.py --suite=suites/example_suite.yaml --collection=ivf_random_l2 --searchonly` , the QPS test will perform again.
3. You can test specified search-parameter by invoke `python run.py --suite=suites/example_suite.yaml --collection=ivf_random_l2 --searchonly -nq=5 -k=20 --runs=10 --clients=50`.
    - If nq or k is specified, the script will only execute the input parameter combination.

Including your test-suite
========================

1. copy or modify suite/example_suite.yaml

TODO
========================

1. Support ANNS stardand Dataset [sift, glove, mnist...]
2. Collect useful data during the run
3. Display of results is more humane

