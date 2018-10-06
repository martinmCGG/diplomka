'''
Created on 4. 10. 2018

@author: miros
'''
import os
from MakeTable import write_all
from MakeCategories import get_categories
from MakeCategories import is_file



def make_matrices(folder='.'):
    os.chdir(folder)
    categories = get_categories('/home/krabec/models/MVCNN/modelnet40v1')
    files = [x for x in os.listdir(folder) if is_file(x,'txt')]
    for file in files:
        misses = count_misses(file, categories)
        make_matrix(misses, categories, file)
        
def make_matrix(misses,categories,file):
    outfile = file.split('.')[0] + '.html'
    with open(outfile, 'w') as out:
        write_all(out,['<!DOCTYPE html>','<html>','<body>','<table style="width:100%">',])
        
        write_row(out, ['Truth/Predicted'] + categories, separator = 'th')
        
        for i in range(len(categories)):
            misses_i = [misses[x] for x in misses.keys() if x[0] == categories[i]]
            write_row(out, [categories[i]] + misses_i, 'td')
        
        write_all(out,['</table>','</body>','</html>'])

def write_row(file, column, separator = 'td'):
    color = 'white'
    file.write('<tr>\n')
    file.write('<{0} bgcolor=\"{2}\">{1}</{0}>'.format('th', column[0], color))
    for i in range(1,len(column)):
        if separator == 'td' and int(column[i]) != 0:
            color = 'red'
        else:
            color = 'white'
        file.write('<{0} bgcolor=\"{2}\">{1}</{0}>'.format(separator, column[i], color))
        
    file.write('</tr>\n')       
    
    
def count_misses(file, categories):
    misses = {}
    for cat1 in categories:
        for cat2 in categories:
            misses[(cat1,cat2)] = 0
    with open(file, 'r') as f:
        f.readline()
        for line in f:
            splited = line.split()
            truth = splited[0]
            prediction = splited[1]
            if truth != prediction:
                misses[(truth, prediction)] +=1
    return misses
        



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to data") 
    
    args = parser.parse_args()
    os.chdir(args.folder)
    make_matrices()
    
if __name__ == '__main__':
    main()