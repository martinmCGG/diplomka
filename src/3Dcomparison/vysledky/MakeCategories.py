import os

def is_file(file,extension):
    splited = file.split('.')
    if len(splited) > 0:
        if splited[-1] == extension:
            return True
    return False

def get_categories(dataset_dir):
    with open(os.path.join(dataset_dir, 'cat_names.txt'), 'r') as f:
        cats = [x.strip() for x in f.readlines()]
    return cats

def make_categories(dataset_dir,outfile, preds, labels, name):
    cats = get_categories(dataset_dir)
    with open(outfile, 'w') as f:
        f.write(name + '\n')
        for i in range(len(preds)):
            f.write("{} {}\n".format(cats[labels[i]], cats[preds[i]]) ) 
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to data") 
    print(os.getcwd())
    args = parser.parse_args()
    os.chdir(args.folder)
    make_categories(os.getcwd())
    
if __name__ == '__main__':
    main()