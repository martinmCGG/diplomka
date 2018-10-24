'''
Created on 5. 5. 2018
We suppose that the images are sorted by categories in separate folders and we are ignoring subfolders
@author: Mirek
'''
import os
import argparse
    
def prepare_category(index, folder, list_folder, args):
    allfiles = lift_from_subdirectories(folder)
    allmodels = list(set(["_".join(file.split('_')[:-1]) for file in allfiles]))
    
    valsize = int(args.val * len(allmodels))
    testsize = int(args.test * len(allmodels))
    
    valmodels = allmodels[0:valsize]
    create_text_files("val",  list_folder, index, valmodels, args.views)
    
    testmodels = allmodels[valsize: valsize+testsize]
    create_text_files("test", list_folder, index, testmodels, args.views)
    
    trainmodels = allmodels[valsize+testsize:]
    create_text_files("train",list_folder, index, trainmodels, args.views)
    


def create_text_files(dataset, list_folder, index, models, views):
    allmodels = models
    with open (os.path.join(list_folder,dataset + "_list.txt"), "a") as list_of_lists:
        for model in allmodels:
            model_file_name = model + "off.txt"
            print("{0} {1}".format(model_file_name,index), file=list_of_lists)
            with open (model_file_name, "w") as model_file:
                print(index, file = model_file)
                print(views,file = model_file)
                for i in range(views): 
                    print(os.path.join("{0}_{1:0>3}.jpg".format(model,i+1)), file=model_file)
            
            

def lift_from_subdirectories(folder):
    allfiles = []
    newfiles = []
    for o in os.walk(folder):
        if o[2]:
            for file in o[2]:
                allfiles.append(os.path.join(o[0],file))
    for file in allfiles:
        name = os.path.join(folder , os.path.basename(file))
        os.rename(file, name)
        newfiles.append(name)
    return newfiles
    
def make_one_file(category,category_number,number, args, dataset):
    outfile = os.path.join(args.outfolder, "{0}_{1:0>3}.txt".format(category, number))
    with open(outfile,'w') as f:
        print(category_number, file = f)
        print(args.views,file = f)
        for i in range(args.views): 
            print("{0}/{1}/{2}/{1}_{3:0>4}_{4:0>3}.png".format(args.folder,category,dataset,number,i+1), file=f)
        
    return outfile

def make_all_files(args):
    categories = sorted(os.listdir(args.folder))
    
    for i in range(len(categories)):
        offset = 0
        for dataset in ['train','test']:
            aggfile = '{}/{}files.txt'.format(args.outfolder, dataset)
            samples = len(os.listdir('{}/{}/{}'.format(args.folder,categories[i],dataset))) //   args.views
            with open(aggfile, 'a') as f:
                for j in range(offset,samples+offset):
                    print('{} {}'.format(make_one_file(categories[i],i, j+1, args, dataset), i),file=f)
                    
            offset += samples
        
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default='/home/krabec/models/ROTNET/ModelNet40v2/modelnet40v2png_ori4', type=str, help="root folder of all the images")
    parser.add_argument("--outfolder", default='/home/krabec/models/MVCNN/datan', type=str, help="out folder")
    parser.add_argument("--views", default=20, type=int, help="number of views per model")
    args = parser.parse_args()
    
    folder = args.folder
    make_all_files(args)
        
if __name__ == "__main__":
    main()