"""Diagnose pretrained weight loading — check what's loaded vs missing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.bert import BertConfig, BertModel, load_pretrained_weights, _PARAM_MAPPING, _build_encoder_mapping


def main():
    config = BertConfig()
    model = BertModel(config)
    our_state = model.state_dict()

    mapping = {**_PARAM_MAPPING}
    mapping.update(_build_encoder_mapping(config.num_hidden_layers))

    from transformers import BertModel as HFBertModel
    hf_model = HFBertModel.from_pretrained("bert-base-uncased")
    hf_state = hf_model.state_dict()

    loaded, missing, shape_mismatch = [], [], []
    for our_name, param in our_state.items():
        hf_name = mapping.get(our_name)
        if hf_name is None or hf_name not in hf_state:
            missing.append(our_name)
        elif param.shape != hf_state[hf_name].shape:
            shape_mismatch.append((our_name, param.shape, hf_state[hf_name].shape))
        else:
            loaded.append(our_name)

    total = len(our_state)
    print(f"\n=== Weight Loading Diagnosis ===")
    print(f"Total parameters:  {total}")
    print(f"Loaded from HF:    {len(loaded)}")
    print(f"Missing (random):  {len(missing)}")
    print(f"Shape mismatch:    {len(shape_mismatch)}")
    print(f"Load rate:         {len(loaded)/total*100:.1f}%")

    if missing:
        print(f"\nMissing parameters:")
        for name in missing:
            print(f"  - {name}")

    if shape_mismatch:
        print(f"\nShape mismatches:")
        for name, ours, hf in shape_mismatch:
            print(f"  - {name}: ours={ours}, hf={hf}")

    # Also check: are there HF params we're NOT using?
    used_hf = set(mapping.values())
    unused_hf = [k for k in hf_state.keys() if k not in used_hf]
    if unused_hf:
        print(f"\nUnused HF parameters ({len(unused_hf)}):")
        for name in unused_hf:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
