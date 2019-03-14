from __future__ import print_function
import sys
from config import get_config

if __name__ == "__main__":
    config = get_config()
    if config.dataset_type == 'modelnet':
        from lib.processors.modelnet40 import prepare
    elif config.dataset_type == 'shapenet':
        from lib.processors.shapenet import prepare
    else:
        print("Unsupported dataset")
        sys.exit(1)
    with open (config.log_file,'w') as f:
        prepare(config.data, config.output, f)