import json
from pathlib import Path
from typing import Dict


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


def _update_metrics_dict(metrics_file: Path, metrics_dict: Dict):
    existing_metrics_dict = {}
    if metrics_file.exists():
        with open(metrics_file, "r") as fp:
            existing_metrics_dict = json.load(fp)

    with open(metrics_file, "w") as fp:
        json.dump(
            {**existing_metrics_dict, **metrics_dict},
            fp,
            default=dumper,
            sort_keys=True,
            indent=4,
        )
