import os, random, shutil

val_dir = '../VOC/images/val_voc/'
detect_dir = '../VOC/images/detect500/'
# val_dir = '../coco/images/val2017/'
# detect_dir = '../coco/images/detect500/'
if not os.path.exists(detect_dir):
    os.makedirs(detect_dir)

filenames = random.sample(os.listdir(val_dir), 500)
for fname in filenames:
    srcpath = os.path.join(val_dir, fname)
    shutil.copy(srcpath, detect_dir)