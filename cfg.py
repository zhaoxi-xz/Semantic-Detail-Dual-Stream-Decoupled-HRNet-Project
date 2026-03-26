BATCH_SIZE = 2
EPOCH_NUMBER =200
DATASET = ['Greenhouse3', 3]

SEGFORMER_VERSION = 'B2'

crop_size = (512, 512)

class_dict_path = './Datasets/' + DATASET[0] + '/class_dict.csv'
TRAIN_ROOT = './Datasets/' + DATASET[0] + '/train/images'
TRAIN_LABEL = './Datasets/' + DATASET[0] + '/train/masks'
VAL_ROOT = './Datasets/' + DATASET[0] + '/val/images'
VAL_LABEL = './Datasets/' + DATASET[0] + '/val/masks'
TEST_ROOT = './Datasets/' + DATASET[0] + '/test/images'
TEST_LABEL = './Datasets/' + DATASET[0] + '/test/masks'
