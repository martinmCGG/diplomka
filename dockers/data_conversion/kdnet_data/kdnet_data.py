from __future__ import print_function
import sys
from config import get_config

if __name__ == "__main__":
    config = get_config()
    if config.dataset_type == 'modelnet':
        from lib.processors.modelnet40 import prepare
    elif config.dataset_type == 'shapenet':
        from lib.processors.shapenetcore import prepare
    else:
        print("Unsupported dataset")
        sys.exit(1)
    prepare(config)