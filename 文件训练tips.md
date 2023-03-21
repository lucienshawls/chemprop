# 文件训练tips

## 代码运行前准备

进入COVIDVS-master文件夹（桌面hyx文件夹里面稍微找下就可以）

注意后面所用的文件路径绝对路径和相对路径都可以，如果不确定相对路径直接复制直接路径

```
conda activate chemprop
```

## 训练超参数

```
python hyperparameter_optimization.py --gpu 0 --data_path ../data/traindata.csv --features_path ../data/traindata-feat.npy --no_features_scaling --dataset_type classification  --num_iters 20 --config_save_path hyperopt_it20.json 
```

## 训练

```
python train.py --gpu 0 --data_path ./dataset/traindata.csv --features_path ./dataset/traindata-feat.npy --no_features_scaling --save_dir covidvs1/ --dataset_type classification --split_sizes 0.9 0.1 0.0 --num_folds 20 --config_path hyperopt_it20.json 
```

--data_path 训练集位置

--features_path 训练集对应feat的位置

 --save_dir 保存模型的路径

--split_sizes 如何划分训练集我一般0.8，0.1，0.1

--num_folds 做多少折交叉验证

--config_path 调超参数所对应的json文件位置

## 迁移

```
python finetune.py --gpu 0 --data_path ../data/finetunev1.csv --features_path ./dataset/finetunev1-feat.npy --save_dir covidvs2/ --checkpoint_path covidvs1/fold_0/model_0/model.pt --split_sizes 0.9 0.1 0.0 --config_path hyperopt_it20.json --dataset_type classification --init_lr 1e-4 --batch_size 20 --epochs 30
```

--checkpoint_path 预训练模型的存放位置（注意此处的covidvs1/为之前预训练模型对应的存放位置后面的/fold_0/model_0/model.pt不能省略，fold_0为选择交叉验证的第几折模型）

--init_lr 学习率，就是迁移后模型在原模型基础上调整的快慢，具体一般就用1e-4

--batch_size 一次训练所抓取的数据样本数量，越大越准确但是需要较长训练时间，过大也不会提高模型准确性

--epochs 迭代次数，也是越大越准确但是需要较长训练时间

## 预测

```
python predict.py --gpu 0 --test_path ./dataset/launched.csv --features_path ./dataset/launched-feat.npy --preds_path preds_covidvs1_launched.csv --checkpoint_dir covidvs1/ --use_compound_names
```

--test_path 预测集位置

--features_path 预测集对应的feat文件位置

--preds_path 预测结果位置

--checkpoint_dir  用于预测的模型位置