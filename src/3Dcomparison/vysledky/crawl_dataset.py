import os

def load_canonical(folder):
    canonical = {}
    os.chdir(folder)
    
    cats = os.listdir('.')
    for cat in cats:
        canonical[cat] = {}
        _load_dir('test', cat, canonical)

def _load_dir(dataset, cat, canonical):
    canonical[cat][dataset] = []
    print(os.listdir(os.path.join(cat,dataset)))
    files = os.listdir(os.path.join(cat,dataset))
    for file in files:
        canonical[cat][dataset].append(file.split('_')[1].split('.')[0])
    print(canonical[cat][dataset])   
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelnet", default=".", type=str, help="Path to data") 
    
    args = parser.parse_args()
    
if __name__ == '__main__':
    main()