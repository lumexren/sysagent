import tempfile
from pathlib import Path
import sysagent.local_mind as lm

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    orig_ds, orig_m, orig_w, orig_l = lm.DATASET_DIR, lm.MODELS_DIR, lm.WEIGHTS_FILE, lm.LORA_TRAIN_FILE
    try:
        lm.DATASET_DIR = tmp_path / 'dataset'
        lm.MODELS_DIR = tmp_path / 'models'
        lm.WEIGHTS_FILE = lm.MODELS_DIR / 'local_mind_weights.json'
        lm.LORA_TRAIN_FILE = lm.DATASET_DIR / 'lora_chatml.txt'
        mind = lm.LocalMind()
        assert isinstance(mind.weights, dict) and 'tool_centroids' in mind.weights
        tool, _, score = mind.predict_tool('check system status')
        assert tool == 'system_status' and score > 0
    finally:
        lm.DATASET_DIR, lm.MODELS_DIR, lm.WEIGHTS_FILE, lm.LORA_TRAIN_FILE = orig_ds, orig_m, orig_w, orig_l