import os

def is_file(file,extension):
    splited = file.split('.')
    if len(splited) > 0:
        if splited[-1] == extension:
            return True
    return False

def get_categories(folder):
    cats = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder,x))]
    return sorted(cats)

def make_categories(folder, outfile, preds, labels, name):
    cats = get_categories(folder)
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