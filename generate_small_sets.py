import os, random, shutil

# all_dir = '../VOC/labels/train_voc/'
# small_dir = '../VOC/labels/train_voc_25/'
all_dir = '../VOC/images/train_voc/'
small_dir = '../VOC/images/train_voc_50/'
if not os.path.exists(small_dir):
    os.makedirs(small_dir)

num_label = len(os.listdir(all_dir))
filenames = random.sample(os.listdir(all_dir), round(0.5*num_label))
for fname in filenames:
    srcpath = os.path.join(all_dir, fname)
    shutil.copy(srcpath, small_dir)