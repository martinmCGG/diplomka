import os


def replace(file, what, for_what):
    # Read in the file
    with open(file, 'r') as f:
        filedata = f.read()

    # Replace the target string
    filedata = filedata.replace(what, for_what)
    #print(filedata)
    # Write the file out again
    with open(file, 'w') as f:
        f.write(filedata)
        
def set_num_cats(file, num_cats, views):
    replace(file, '$NUMCATS', str(num_cats+1))
    replace(file, "$INNER", str(num_cats+1 * views))
    
def set_batch_size(file, batch_size):
    replace(file, "$BATCHSIZE", str(batch_size))
    