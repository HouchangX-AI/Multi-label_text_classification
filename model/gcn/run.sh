mkdir data/corpus

python generate_datafile.py

python build_graph.py baidu_95

python train.py baidu_95 True