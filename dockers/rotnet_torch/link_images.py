import os

def link_images(file, name, out_dir, num_cats=40, VIEWS=12):
    
    out_dir = os.path.join(out_dir, name)
    os.system("mkdir -m 777 {}".format(out_dir))
    for i in range(num_cats):
        os.system("mkdir -m 777 {}".format(os.path.join(out_dir,str(i))))

    with open(file, "r") as f:
        for line in f:
            line2 = line.split()[0]
            with open(line2, "r") as f2:
                cat = f2.readline().strip()
                f2.readline().strip()
                for view in range(VIEWS):
                    what = f2.readline().strip()
                    where = os.path.join(out_dir, cat, os.path.basename(what))
                    print(what, where)
                    cmd = "ln -s {} {}".format(what, where)
                    os.system(cmd)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", type=str, help="Path list of train images")
    parser.add_argument("test_file",type=str, help="Path list of test images")
    parser.add_argument("--out", type=str, default="./RotData", help="Path to output directory")
    parser.add_argument("--views",type=int, default=12, help="Number of views")
    
    args = parser.parse_args()
    os.system("mkdir -m 777 {}".format(args.out))
    link_images(args.train_file, "train", args.out)
    link_images(args.test_file, "test", args.out)
    cmd = "ln -s {} {}".format(os.path.join(args.out, "test"), os.path.join(args.out, "val"))
    os.system(cmd)
    
    
if __name__ == '__main__':
    main()