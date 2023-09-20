## 项目简介

```bash
│  .gitignore   # gitignore文件, 不保存dataset和checkpoint文件夹
│  model.py     # 模型配置类和模型类
│  README.md    # 简介
│  sample.py    # 生成文本用的脚本, 可运行
│  train.py     # 训练模型用的脚本, 可运行
│  utils.py     # 加载训练集
│  vocab.py     # 词典, 及其read和save操作
├─checkpoint    # 用于存放模型的checkpoint
├─dataset       # 用于存放缓存的数据集文本: train.txt, test.txt和字典: vocab.json
```

## 项目运行

- 要训练项目, 可以直接运行train.py
- 要加载训练好的模型并生成文本, 可以直接运行sample.py