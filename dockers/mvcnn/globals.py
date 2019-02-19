"""
constants for the data set.
ModelNet40 for example
"""
NUM_CLASSES = 40
NUM_VIEWS = 12
TRAIN_LOL = '/data/converted/train.txt'
VAL_LOL = '/data/converted/test.txt'
TEST_LOL = '/data/converted/test.txt'


"""
constants for both training and testing
"""
BATCH_SIZE = 64

# this must be more than twice the BATCH_SIZE
INPUT_QUEUE_SIZE = 4 * BATCH_SIZE

TRAIN_FOR = 100
"""
constants for training the model
"""
INIT_LEARNING_RATE = 0.0001

# save the progress to checkpoint file every SAVE_PERIOD iterations
# this takes tens of seconds. Don't set it smaller than 100.
SAVE_PERIOD = 10

