from __future__ import print_function
import os


def get_categories(dataset_dir):
    with open(os.path.join(dataset_dir, 'cat_names.txt'), 'r') as f:
        cats = [x.strip() for x in f.readlines()]
    return cats

def write_all(file,strings):
    for s in strings:
        file.write(s + '\n')    

def is_file(file,extension):
    splited = file.split('.')
    if len(splited) > 0:
        if splited[-1] == extension:
            return True
    return False

def write_eval_file(dataset_dir,outfile, preds, labels, name):
    cats = get_categories(dataset_dir)
    with open(outfile, 'w') as f:
        f.write(name + '\n')
        for i in range(len(preds)):
            f.write("{} {}\n".format(cats[labels[i]], cats[preds[i]]) ) 

def make_table(out, dataset_dir='.', folder='.'):
    categories = get_categories(dataset_dir)
    files = collect_files(folder)
    with open(out, 'w') as f:
        write_all(f,['<!DOCTYPE html>','<html>','<body>','<table style="width:100%">',])
        write_row(f, ['Model/Cat_Acc'] + categories, separator = 'th')
        
    for file in files:
        misses, counts, name  = count_misses(file, categories)
        counts = [counts[cat] for cat in categories]
        class_accs = get_class_acs(misses, counts, categories)
        make_subtable(class_accs, out, name)
        
    with open(out, 'a') as f:
        write_row(f, (len(categories)+1)*[' '], separator = 'td')
        write_row(f, ['Model', 'Accuraccy', 'Avg Class Acc'] + (len(categories)-1)*[' '], separator = 'th')
        
    for file in files:
        misses, counts, name = count_misses(file, categories)
        counts = [counts[cat] for cat in categories]
        class_accs = get_class_acs(misses, counts, categories)
        add_accuracies_to_table(misses, counts, categories, class_accs, out, name)
        
    with open(out,'a') as f:
        write_all(f,['</table>','</body>','</html>'])
       

def add_accuracies_to_table(misses, counts, categories,class_accs,out, name):

    accuracy = round(count_accuracy(misses, counts, categories),2)
    avg_class_acc = round(sum(class_accs)/len(categories),2)
    
    with open(out,'a') as f:
        write_row(f, [name, accuracy, avg_class_acc], separator = 'td')
         
def make_subtable(class_accs, out, name):
    with open(out,'a') as f:
        write_row(f, [name] + [round(x,2) for x in class_accs] , separator = 'td')
    return class_accs
    

def get_class_acs(misses, counts, categories):
    mistakes = [0] * len(categories)
    for i in range(len(categories)):
        cats = [misses[x] for x in misses.keys() if x[0] == categories[i]]
        mistakes[i] = sum(cats)
    if type(counts) is dict:
        counts = list(counts.values())
    class_accs = [(counts[i] - mistakes[i]) / counts[i] *100 for i in range(len(categories))]
    return class_accs

def make_matrix(dataset_dir, file, outdir):
    categories = get_categories(dataset_dir)
    outfile = os.path.basename(file).split('.')[0] + '.html'
    outfile = os.path.join(os.path.dirname(file),outfile)
    misses, counts, name = count_misses(file, categories)
    
    with open(outfile, 'w') as out:
        write_all(out,['<!DOCTYPE html>','<html>','<body>','<table style="width:100%">',])
        write_row(out, ['Truth/Predicted'] + categories, separator = 'th')
        for i in range(len(categories)):
            misses_i = [misses[x] for x in misses.keys() if x[0] == categories[i]]
            write_row(out, [categories[i]] + misses_i, 'td', colors = True)
        
        write_all(out,['</table>','</body>','</html>'])


def write_row(file, column, separator = 'td', colors = False):
    color = 'white'
    file.write('<tr>\n')
    file.write('<{0} bgcolor=\"{2}\">{1}</{0}>'.format('th', column[0], color))
    for i in range(1,len(column)):
        if not colors:
            color = 'white'
        elif int(column[i]) != 0:
            color = 'red'
        else:
            color = 'white'
        file.write('<{0} bgcolor=\"{2}\">{1}</{0}>'.format(separator, column[i], color))
        
    file.write('</tr>\n')       
    
def count_accuracy(misses, counts, categories):
    mistakes = [0]*len(categories)
    for i in range(len(categories)):
        cats = [misses[x] for x in misses.keys() if x[0] == categories[i]]
        mistakes[i] = sum(cats)
    return 100 * (sum(counts)- sum(mistakes)) / sum(counts)
    
    
def count_misses(file, categories):
    misses = {}
    counts = {}
    for cat1 in categories:
        counts[cat1] = 0
        for cat2 in categories:
            misses[(cat1,cat2)] = 0
    with open(file, 'r') as f:
        name = f.readline()
        for line in f:
            splited = line.split()
            truth = splited[0]
            counts[truth]+=1
            prediction = splited[1]
            if truth != prediction:
                misses[(truth, prediction)] +=1
    return misses, counts, name

def is_correct_file(file):
    with open(file,'r') as f:
        content = f.readlines()
        if len(content) == 0:
            return False
        for line in content[1:]:
            splited = line.split()
            if len(splited)!=2:
                return False
    return True
    
def collect_files(dir):
    all_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.txt'):
                all_files.append(os.path.join(root,file))
    
    all_files = [file for file in all_files if is_correct_file(file)]
    return all_files
    

def sort_dict_by_value(d, reverse=True):
    return [(k, d[k]) for k in sorted(d, key=d.get, reverse=reverse)]

def get_bad(args):
    
    with open(args.out, 'w') as f:
        print("Bad categories: ", file=f)
        print("", file=f)
    
    def print_one_file(name, misses, counts, class_accs, categories, file, top=10):
        
        with open(file, 'a') as f:
            print(name, file=f)
            bad_cats = {}
            for i in range(len(class_accs)):
                bad_cats[categories[i]] = class_accs[i]
            bad_cats = sort_dict_by_value(bad_cats, reverse=False)
            for bad in bad_cats[0:top]:
                what = bad[0]
                percentage = bad[1]
                print("{}{}{:.2f}%     ({} cases)".format(what, (14-len(what))*" ", percentage, counts[what]), file=f)
            
            bad_pairs = misses
            for key in bad_pairs.keys():
                bad_pairs[key] = bad_pairs[key] / counts[key[0]] * 100 
            bad_pairs = sort_dict_by_value(misses)
            
            for bad in bad_pairs[0:top]:
                what = bad[0][0]
                for_what = bad[0][1]
                percentage = bad[1]
                print("{}{}->{}{}     {:.2f}%     ({} cases)".format(what, (14-len(what))*" ", (14-len(for_what))*" ", for_what, percentage, counts[what]), file=f)
            print("", file=f) 
        
    files = collect_files(args.folder)
    categories = get_categories(args.dataset)
    all_class_accs = len(categories) * [0]
    all_bad_pairs = {}
    for cat1 in categories:
        for cat2 in categories:
            all_bad_pairs[(cat1,cat2)] = 0
          
    for file in files:
        misses, counts, name = count_misses(file, categories)
        class_accs = get_class_acs(misses, counts, categories)
        
        all_class_accs = [all_class_accs[i] + class_accs[i] for i in range(len(categories))]
        for key in all_bad_pairs.keys():
            all_bad_pairs[key] += misses[key]
        
        print_one_file(name, misses, counts, class_accs, categories, args.out)
    
    for key in all_bad_pairs.keys():
        all_bad_pairs[key] /= len(files)
    all_class_accs = [x/len(files) for x in all_class_accs]
        
    print_one_file("AVERAGE", all_bad_pairs, counts, all_class_accs, categories, args.out, top=40)
        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to data")
    parser.add_argument("--out", default="./bad_cats.txt", type=str, help="Path to output file") 
    parser.add_argument("--dataset", default='.',help="Path dataset containg file cat_names.txt")
    
    args = parser.parse_args()
    get_bad(args)
    make_table("./table.html", dataset_dir = args.dataset, folder = args.folder)
    
    
if __name__ == '__main__':
    main()