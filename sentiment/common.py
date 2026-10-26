"""Settings shared by every script: which base model and dataset, pinned to exact revisions so runs are reproducible."""

BASE = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"

DATASET = "lansinuote/ChnSentiCorp"
DATASET_REVISION = "b0c4c119c3fb33b8e735969202ef9ad13d717e5a"
LABELS = ["negative", "positive"]          # the dataset's ClassLabel order: 0 = negative, 1 = positive

DATA_DIR = "data"
RUNS_DIR = "runs"
