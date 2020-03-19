Benchmarking Milvus QPS Performance
==============================
This project contains some tools to benchmark various implementations of ANNS algorithm already supported in milvus

Install
=======

The only prerequisite is Python and Milvus (tested with 3.6.8).

1. Clone the repo.
2. Run `pip install -r requirements.txt`.

Running
=======

1. Run `python run.py` (this can take an extremely long time)

You can customize the algorithms and datasets if you want to:

* Check that `example_suite.yaml` contains the parameter settings that you want to test
* To run specified suite, invoke `python run.py --suite=suites/your_own_suite.yaml `. See `python run.py --help` for more information on possible settings. Note that experiments can take a long time. 
* To search specifing collection and skip the train stage(make sure the collection is already created and trained).  An example call: `python run.py --suite=suites/example_suite.yaml --searchonly --collection=example_collection `. 


Including your test-suite
========================

1. copy or modify suite/example_suite.yaml


TODO
========================

1. Support ANNS stardand Dataset [sift, glove, mnist...]
2. Collect useful data during the run
3. Display of results is more humane