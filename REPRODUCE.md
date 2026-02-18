### 📌 Step 1. 폴더 생성

```bash
mkdir -p DATA/medical_diff_vqa_processed
mkdir -p DATA/no_rg
touch DATA/no_rg/TRAIN_SAMPLES_NO_rg.txt
touch DATA/no_rg/TEST_SAMPLES_NO_rg.txt
```

---

### 📌 Step 2. train/val/test CSV 생성

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

---

### 📌 Step 3. validation split 통일

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

### 📌 Step 4. vocab 생성

```bash
python << 'PY'
import pandas as pd, re
from collections import Counter

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

with open(f"{root}/vocab_diff.tgt","w") as f:
    for w,_ in counter.most_common():
        f.write(w+"\n")

print("Vocab generated.")
PY
```

---

### 📌 Step 5. `Vision_Encoder_Decoder_MDiffVQA/paths.py` 수정

```python
IMAGES_MIMIC_PATH = "/data2/local_datasets/ameer_data/mimic-cxr-jpg/2.0.0/files"

DICT_CSV_MIMIC_PATH = {
    "train": "DATA/medical_diff_vqa_processed/medical_vqa_pair_onlydiffquestions_train.csv",
    "validation": "DATA/medical_diff_vqa_processed/medical_vqa_pair_onlydiffquestions_val.csv",
    "test": "DATA/medical_diff_vqa_processed/medical_vqa_pair_onlydiffquestions_test.csv",
}

VOCAB_PATH = "DATA/medical_diff_vqa_processed/vocab_diff.tgt"

PATH_IDS_NO_RG_TRAIN = "DATA/no_rg/TRAIN_SAMPLES_NO_rg.txt"
PATH_IDS_NO_RG_TEST = "DATA/no_rg/TEST_SAMPLES_NO_rg.txt"
```