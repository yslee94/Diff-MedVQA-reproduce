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
    "train": "/data/yeseul/projects/diff-VQA/Diff-MedVQA/DATA/medical_diff_vqa_processed/medical_vqa_pair_onlydiffquestions_train.csv",
    "validation": "/data/yeseul/projects/diff-VQA/Diff-MedVQA/DATA/medical_diff_vqa_processed/medical_vqa_pair_onlydiffquestions_val.csv",
    "test": "/data/yeseul/projects/diff-VQA/Diff-MedVQA/DATA/medical_diff_vqa_processed/medical_vqa_pair_onlydiffquestions_test.csv",
}

VOCAB_PATH = "/data/yeseul/projects/diff-VQA/Diff-MedVQA/DATA/medical_diff_vqa_processed/vocab_diff.tgt"

PATH_IDS_NO_RG_TRAIN = "/data/yeseul/projects/diff-VQA/Diff-MedVQA/DATA/no_rg/TRAIN_SAMPLES_NO_rg.txt"
PATH_IDS_NO_RG_TEST = "/data/yeseul/projects/diff-VQA/Diff-MedVQA/DATA/no_rg/TEST_SAMPLES_NO_rg.txt"
```

---

### 📌 Step 6. Stage2 프로젝트 루트로 이동

```bash
cd Vision_Encoder_Decoder_MDiffVQA
```

---

### 📌 Step 7. CSV 경로가 제대로 읽히는지 확인 (paths.py 체크)

```bash
python - << 'PY'
import pandas as pd
from paths import DICT_CSV_MIMIC_PATH, IMAGES_MIMIC_PATH

print("IMAGES_MIMIC_PATH =", IMAGES_MIMIC_PATH)
print("DICT_CSV_MIMIC_PATH =", DICT_CSV_MIMIC_PATH)

for split, path in DICT_CSV_MIMIC_PATH.items():
    df = pd.read_csv(path)
    print(f"[{split}] rows={len(df)} cols={df.columns.tolist()}")
    print(df.head(2)[["study_id","subject_id","ref_id","question_type","question","answer","split"]])
PY
```

---

### 📌 Step 8. dataset 클래스 import 확인 (mimic_Dataset)

```bash
python - << 'PY'
from mydatasets.mimic_dataset import mimic_Dataset
print("dataset class:", mimic_Dataset)
PY
```

---

### 📌 Step 9. 가장 안전한 “sanity run” 실행 (가중치 로드 없이)

```bash
python train/mytrain_nll.py --help | head -n 120
```

