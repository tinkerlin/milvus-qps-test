随机dataset 使用固定种子生成
只要传了 collection-name 就跳过train 直接 search 即可

python qps-test.py --suits example_suits.yaml # 执行对应数据集
python qps-test.py --dataset sift1m --collection_name sift-ivf -nq 1 -topk 10 -params 32 # 直接运行，只跑这一组参数
python qps-test.py --dataset sift1m --collection_name sift-ivf --suits example_suits # 找到对应yaml文件的collection, 获取对应的search-params 再执行


# support params
params:
  mode: default single
  dataset:
  suits:
  corrency_num: defalut 1
  run_count: (only search)default 5
  nq:
  topk:
  params:

Directory
- data
  - 1000k-L2.dataset
- log
  - year-month-day-time.log
- result
    - collection-name
      - time-result
        - Meta
        - result_raw
        - report
- TestMe
  - main.py # params/entry
  - runner.py # execute
  - result.py # analysis/read/write result
  - data.py # gen/read/write result
  - utils.py # balabala
  - ann-helper.py # recall/qps

QPS 的metric
多client下，计算各自1秒内的执行次数
但client下同

单client 计算平均rt
分别计算各自的qps然后累加
