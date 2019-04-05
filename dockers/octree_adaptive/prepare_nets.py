import os


def replace(file, what, for_what):
    # Read in the file
    with open(file, 'r') as f:
        filedata = f.read()

    # Replace the target string
    filedata = filedata.replace(what, for_what)

    # Write the file out again
    with open(file, 'w') as f:
        f.write(filedata)
        
def set_num_cats(file, num_cats):
    replace(file, '$NUMCATS', str(num_cats+1))