'''
Created on 4. 10. 2018

@author: miros
'''
import os
from MakeTable import write_all
from MakeCategories import get_categories
from MakeCategories import is_file


def make_matrices(out,folder='.'):
    os.chdir(folder)
    categories = get_categories('/home/krabec/models/MVCNN/modelnet40v1')
    files = [x for x in os.listdir(folder) if is_file(x,'txt')]
    with open(out, 'w') as f:
        write_all(f,['<!DOCTYPE html>','<html>','<body>','<table style="width:100%">',])
        write_row(f, ['Model/Cat_Acc'] + categories, separator = 'th')
        
    for file in files:
        misses, counts = count_misses(file, categories)
        counts = [counts[cat] for cat in categories]
        make_matrix(misses, categories, file)
        class_accs = get_class_acs(misses, counts, categories)
        make_table(class_accs,out, file)
        
    with open(out, 'a') as f:
        write_row(f, 41*[' '], separator = 'td')
        write_row(f, ['Model', 'Accuraccy', 'Avg Class Acc'] + 39*[' '], separator = 'th')
        
    for file in files:
        misses, counts = count_misses(file, categories)
        counts = [counts[cat] for cat in categories]
        class_accs = get_class_acs(misses, counts, categories)
        add_accuracies_to_table(misses, counts, categories, class_accs, out, file.split('.')[-2])
        
    with open(out,'a') as f:
         write_all(f,['</table>','</body>','</html>'])
         
    
def add_accuracies_to_table(misses,counts, categories,class_accs,out, name):

    accuracy = round(count_accuracy(misses, counts, categories),2)
    avg_class_acc = round(sum(class_accs)/len(categories),2)
    
    with open(out,'a') as f:
         write_row(f, [name, accuracy, avg_class_acc], separator = 'td')
         
def make_table(class_accs, out, name):
    
    with open(out,'a') as f:
        write_row(f, [name] + [round(x,2) for x in class_accs] , separator = 'td')
    return class_accs
    

def get_class_acs(misses, counts, categories):
    mistakes = [0] * len(categories)
    
    for i in range(len(categories)):
        cats = [misses[x] for x in misses.keys() if x[0] == categories[i]]
        mistakes[i] = sum(cats)
    class_accs = [(counts[i] - mistakes[i]) / counts[i] *100 for i in range(len(categories))]
    return class_accs

def make_matrix(misses,categories,file):
    
    outfile = file.split('.')[0] + '.html'
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
        f.readline()
        for line in f:
            splited = line.split()
            truth = splited[0]
            counts[truth]+=1
            prediction = splited[1]
            if truth != prediction:
                misses[(truth, prediction)] +=1
    return misses, counts



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to data")
    parser.add_argument("-o", default="./table.html", type=str, help="Path to output file") 
    
    args = parser.parse_args()
    os.chdir(args.folder)
    make_matrices(args.o)
    
if __name__ == '__main__':
    main()