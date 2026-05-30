# 数据集下载（国内网络环境）

服务器无法直连 GitHub / Google Drive 时，按以下步骤手动准备数据。

## 一、MOCHEG（约 5–6 GB）

1. 在能科学上网的机器/浏览器打开仓库：
   - <https://github.com/PLUM-Lab/Mocheg>
2. README 里会给出 Google Drive 链接（或在 [Releases](https://github.com/PLUM-Lab/Mocheg/releases) 页面下载发布的 zip 资产）。下载 `mocheg.zip` 或对应资产文件。
3. 用 U 盘 / scp 把文件传到服务器：
   ```bash
   scp mocheg.zip user@server:/mnt/data/yangjun/fact/EACER/data/raw/mocheg/
   ```
4. 解压：
   ```bash
   cd /mnt/data/yangjun/fact/EACER/data/raw/mocheg
   unzip mocheg.zip
   ```
5. 期望目录：
   ```
   data/raw/mocheg/
   ├── train/
   │   ├── Corpus2.csv
   │   ├── Corpus3.csv
   │   └── img_evidence/
   ├── val/
   └── test/
   ```

## 二、MR2（约 7 GB，含图像）

### 推荐路径：百度 AI Studio（国内直连）

1. 浏览器打开：<https://aistudio.baidu.com/datasetdetail/230144>
2. 注册百度账号 → 加入数据集 → 下载（支持网页内或命令行）。
3. 若使用 AI Studio Notebook，可直接 `! cp /home/aistudio/data/data230144/*.zip ./` 拿到 zip 后下载到本地。
4. 上传至服务器并解压：
   ```bash
   scp MR2.zip user@server:/mnt/data/yangjun/fact/EACER/data/raw/mr2/
   cd /mnt/data/yangjun/fact/EACER/data/raw/mr2
   unzip MR2.zip
   ```

### 备选：Google Drive

文件 ID：`14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4`，链接：
<https://drive.google.com/file/d/14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4/view?usp=sharing>

在能联网的机器上：
```bash
pip install gdown
gdown --id 14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4 -O MR2.zip
unzip MR2.zip -d MR2/
```

### 期望目录
```
data/raw/mr2/
├── dataset_items_train.json
├── dataset_items_val.json
├── dataset_items_test.json
├── img/
└── img_html_news/
```

## 三、数据转换到本项目格式

数据放好后，本项目期望统一 JSONL（README §13）格式。执行：

```bash
# MOCHEG
python scripts/preprocess_mocheg.py \
    --claim_csv data/raw/mocheg/train/Corpus2.csv \
    --evidence_csv data/raw/mocheg/train/Corpus3.csv \
    --out_dir data/processed

# MR2（脚本见 scripts/preprocess_mr2.py，下一步会写）
python scripts/preprocess_mr2.py \
    --items_json data/raw/mr2/dataset_items_train.json \
    --image_root data/raw/mr2 \
    --split train \
    --out_dir data/processed
```

## 四、镜像 / 替代源

如果以上都不通，还可以试：

- **HuggingFace Hub**（部分镜像把 MOCHEG/MR2 重新发布过）：搜索 `mocheg` / `mr2` —— 注意 license 与原版一致
- **Papers with Code**：<https://paperswithcode.com/dataset/mocheg> 列了所有 mirror
- 直接邮件联系作者（MOCHEG: Barry Yao；MR2: Xuming Hu）
