python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision full_supervision --training_type normal --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision full_supervision --training_type normal --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision full_supervision --training_type normal --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 20 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 20 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 20 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 50 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 50 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 50 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 100 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 100 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 100 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 20 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 20 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 20 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 50 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 50 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 50 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 100 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 100 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 100 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss phgce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss phgce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss phgce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss phgce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss phgce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss phgce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss phgce --seed 1
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss phgce --seed 2
python train_dl_model.py --config configs/actionsense/shallow_d.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss phgce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision full_supervision --training_type normal --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision full_supervision --training_type normal --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision full_supervision --training_type normal --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 20 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 20 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 20 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 50 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 50 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 50 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 100 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 100 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --sampling_strategy random --features clip+raft --cluster 100 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 20 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 20 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 20 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 50 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 50 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 50 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 100 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 100 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type few_shot --features clip+raft --cluster 100 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss phgce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss phgce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 20 --samples 1 --loss phgce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss phgce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss phgce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 50 --samples 1 --loss phgce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss ce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss ce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss ce --seed 3

python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss phgce --seed 1
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss phgce --seed 2
python train_dl_model.py --config configs/actionsense/tinyhar.yaml --neptune --supervision weak_labelling --training_type normal --features clip+raft --cluster 100 --samples 1 --loss phgce --seed 3
