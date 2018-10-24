'''
Created on 2. 10. 2018

@author: miros
'''
import os
from MakeCategories import is_file
    

def make_table(outfile,folder='.'):

    columns = parse_column(folder)
    
    with open(outfile, 'w') as out:
        write_all(out,['<!DOCTYPE html>','<html>','<body>','<table style="width:100%">',])
            
        write_row(out, columns, 0, 'th')
        for i in range(1,len(columns[1])):
            write_row(out, columns, i, 'td')
            
        write_all(out,['</table>','</body>','</html>'])
        
def write_row(file, columns, index, separator = 'td'):
    color = 'white'
    file.write('<tr>\n')
    for i in range(len(columns)):
        if separator == 'td' and i>0:
            if columns[0][index] != columns[i][index]:
                color = 'red'
            else:
                color = 'green'
        file.write('<{0} bgcolor=\"{2}\">{1}</{0}>'.format(separator, columns[i][index], color))
    
    file.write('</tr>\n')
    
    
def write_all(file,strings):
    for s in strings:
        file.write(s + '\n')           
    
def parse_columns(folder):
    os.chdir(folder)
    files = [x for x in os.listdir(folder) if is_file(x,'txt')]
    columns = []
    columns.append(parse_first_column(files[0]))
    for file in files:
        columns.append(parse_column(file))
    return columns

def parse_column(file):
    column = []
    with open(file, 'r') as f:
        column.append(f.readline().rstrip('\n'))
        for line in f:
            column.append(line.split()[1])
    return column

def parse_first_column(file):
    column = []
    with open(file, 'r') as f:
        f.readline()
        column.append('Cat/Model')
        for line in f:
            column.append(line.split()[0])
    return column

def print_accuracies(columns):
    for i in range(1, len(columns)):
        all = 0
        correct = 0
        for j in range(1, len(columns[i])):
            if columns[i][j] == columns[0][j]:
                correct += 1
            all +=1
        print('Accuracy of {} : {}'.format(columns[i][0] , correct/all*100))    
  
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=".", type=str, help="Path to data") 
    parser.add_argument("-o", default="./table.html", type=str, help="Path to output file")
    
    args = parser.parse_args()
    os.chdir(args.folder)
    make_table(args.o)
    

if __name__ == "__main__":
    main()