# Diff-MedVQA Stage 2 Reproduction Guide

## 0. Environment

```bash
conda create -n medvqa_env python=3.9
conda activate medvqa_env

pip install torch torchvision
pip install transformers
pip install albumentations
pip install opencv-python-headless==4.7.0.72
pip install einops tqdm pandas pillow
```

---

## 1. Dataset Assumptions

### MIMIC-CXR-JPG

```
/data2/local_datasets/ameer_data/mimic-cxr-jpg/2.0.0/files
```

### Medical-Diff-VQA source CSV

```
/data2/local_datasets/medical_data/medical_diff_vqa/mimic_pair_questions.csv
```

---

## 2. Create Local Working Folders

```bash
cd /data/yeseul/projects/diff-VQA/Diff-MedVQA

mkdir -p DATA/medical_diff_vqa_processed
mkdir -p DATA/no_rg

touch DATA/no_rg/TRAIN_SAMPLES_NO_rg.txt
touch DATA/no_rg/TEST_SAMPLES_NO_rg.txt
```

---

## 3. Generate train / val / test CSV

```bash
python << 'PY'
import pandas as pd

src="/data2/local_datasets/medical_data/medical_diff_vqa/mimic_pair_questions.csv"
out="DATA/medical_diff_vqa_processed"

df=pd.read_csv(src)

df[df["split"]=="train"].to_csv(f"{out}/medical_vqa_pair_onlydiffquestions_train.csv", index=False)
df[df["split"]=="val"].to_csv(f"{out}/medical_vqa_pair_onlydiffquestions_val.csv", index=False)
df[df["split"]=="test"].to_csv(f"{out}/medical_vqa_pair_onlydiffquestions_test.csv", index=False)

print("Saved train/val/test CSV files.")
PY
```

Normalize validation split:

```bash
python << 'PY'
import pandas as pd
p="DATA/medical_diff_vqa_processed/medical_vqa_pair_onlydiffquestions_val.csv"
df=pd.read_csv(p)
df["split"]="validation"
df.to_csv(p, index=False)
print("Updated validation split.")
PY
```

---

## 4. Generate Vocabulary (with required special tokens)

```bash
python << 'PY'
import pandas as pd, re
from collections import Counter
from pathlib import Path

root="DATA/medical_diff_vqa_processed"
train_path=f"{root}/medical_vqa_pair_onlydiffquestions_train.csv"
df=pd.read_csv(train_path)

def tokenize(s):
    s=str(s).lower()
    s=re.sub(r"[^a-z0-9\s\-\+\/\.]", " ", s)
    return [t for t in s.split() if t]

counter=Counter()
for col in ["question","answer"]:
    for text in df[col].astype(str):
        counter.update(tokenize(text))

vocab_path = Path(f"{root}/vocab_diff.tgt")
vocab_path.write_text("\n".join([w for w,_ in counter.most_common()]) + "\n")

# prepend required special tokens
tokens = [t.strip() for t in vocab_path.read_text().splitlines() if t.strip()]
special = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]"]

seen=set()
new=[]
for t in special + tokens:
    if t not in seen:
        new.append(t); seen.add(t)

vocab_path.write_text("\n".join(new) + "\n")

print("Vocab generated:", vocab_path)
PY
```

---

## 5. Add `images` Column to CSV

The dataset requires an `images` column formatted as:

```
<main_image_relpath>,<ref_image_relpath>
```

```bash
python << 'PY'
import os
import pandas as pd

META = "/data2/local_datasets/ameer_data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz"
CSV_DIR = "/data/yeseul/projects/diff-VQA/Diff-MedVQA/DATA/medical_diff_vqa_processed"

splits = {
    "train": os.path.join(CSV_DIR, "medical_vqa_pair_onlydiffquestions_train.csv"),
    "val":   os.path.join(CSV_DIR, "medical_vqa_pair_onlydiffquestions_val.csv"),
    "test":  os.path.join(CSV_DIR, "medical_vqa_pair_onlydiffquestions_test.csv"),
}

meta = pd.read_csv(META, usecols=["subject_id","study_id","dicom_id","ViewPosition"])
meta["ViewPosition"] = meta["ViewPosition"].astype(str)

frontal = meta[meta["ViewPosition"].isin(["PA","AP"])].copy()
if len(frontal) == 0:
    frontal = meta.copy()

frontal = frontal.sort_values(["subject_id","study_id","dicom_id"])
one = frontal.drop_duplicates(["subject_id","study_id"], keep="first")

def relpath(subject_id: int, study_id: int, dicom_id: str) -> str:
    sid = str(int(subject_id))
    st = str(int(study_id))
    pgrp = "p" + sid[:2]
    return f"{pgrp}/p{sid}/s{st}/{dicom_id}.jpg"

one["img_rel"] = one.apply(lambda r: relpath(r["subject_id"], r["study_id"], r["dicom_id"]), axis=1)
mp = dict(zip(list(zip(one["subject_id"].astype(int), one["study_id"].astype(int))), one["img_rel"].tolist()))

def add_images(df: pd.DataFrame) -> pd.DataFrame:
    subj = df["subject_id"].astype(int)
    main_st = df["study_id"].astype(int)
    ref_st  = df["ref_id"].astype(int)

    main_rel = [mp.get((s, st)) for s, st in zip(subj, main_st)]
    ref_rel  = [mp.get((s, st)) for s, st in zip(subj, ref_st)]

    df = df.copy()
    df["main_img"] = main_rel
    df["ref_img"] = ref_rel

    df = df.dropna(subset=["main_img","ref_img"]).copy()
    df["images"] = df["main_img"] + "," + df["ref_img"]
    df = df.drop(columns=["main_img","ref_img"])

    return df

for split, path in splits.items():
    df = pd.read_csv(path)
    if "images" in df.columns:
        df = df.drop(columns=["images"])
    out = add_images(df)
    out.to_csv(path, index=False)
    print("saved", split)

PY
```

---

## 6. Run Stage 2

```bash
cd Vision_Encoder_Decoder_MDiffVQA
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Sanity check:

```bash
python train/mytrain_nll.py --help | head -n 50
```

Start training:

```bash
python train/mytrain_nll.py \
  --exp_name SANITY_RUN \
  --model_arch SwinBERTFinetuned \
  --hnm False
```

Training should start and display:

```
---- Start Training ----
Train Epoch [0/29] Loss: ...
```
