mkdir data
mkdir output

# download bert.ckpt, move to pretrained_model

python data_processor.py

python run_classifier.py
  --task_name=baidu_95 \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=./data/all_labels \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=6.0 \
  --output_dir=./output/epochs6_baidu_95/

# evaluate test
python evaluate_test.py