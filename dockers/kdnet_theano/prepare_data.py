from __future__ import print_function
import sys

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/ModelNet40A", type=str, help="Path to the dataset")
    parser.add_argument("--dataset", default="modelnet40", type=str, help="Type of dataset - shapenet, modelnet40, modelnet10")
    parser.add_argument("--out", default="/data/ModelNet40A_kdnet", type=str, help="Path to converted ")
    args = parser.parse_args()
    
    if args.dataset == 'modelnet40':
        from lib.processors.modelnet40 import prepare
    elif args.dataset == 'modelnet10':
        from lib.processors.modelnet10 import prepare
    elif args.dataset == 'shapenet':
        from lib.processors.shapenet import prepare
    else:
        print("Unsupported dataset")
        sys.exit(1)
            
    prepare(args.data, args.out)