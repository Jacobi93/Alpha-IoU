import os, random
import numpy as np

# change noise
# clean_dir = '../coco/labels/val2017/'
# noisy_dir = '../coco/labels/val2017_noise_0.4/'
clean_dir = '../VOC/labels/val_voc/'
noisy_dir = '../VOC/labels/val_voc_noise2/'
if not os.path.exists(noisy_dir):
    os.makedirs(noisy_dir)

for clean_file in os.listdir(clean_dir):
    with open(clean_dir + clean_file, 'r') as f:
        data = f.readlines()
    table = []
    for i in range(len(data)):
        clean_list = data[i].split(' ')
        num_list = []
        for j in range(len(clean_list)):
            num_list.append(float(clean_list[j]))
        # change noise
        noise = 0.2
        random_w1 = random.uniform(-noise*num_list[3], noise*num_list[3])
        random_h1 = random.uniform(-noise*num_list[4], noise*num_list[4])
        random_w2 = random.uniform(-noise*num_list[3], noise*num_list[3])
        random_h2 = random.uniform(-noise*num_list[4], noise*num_list[4])
        num_list[1] += random_w1
        num_list[2] += random_h1
        num_list[3] += random_w2
        num_list[4] += random_h2
        # make new bbox in the image
        num_list[3] = np.clip(num_list[3], 0.001, 0.999)
        num_list[4] = np.clip(num_list[4], 0.001, 0.999)
        num_list[1] = np.clip(num_list[1], 0.5*num_list[3], 1-0.5*num_list[3])
        num_list[2] = np.clip(num_list[2], 0.5*num_list[4], 1-0.5*num_list[4])
        table.append(num_list)

    noise_anno = open(noisy_dir + clean_file, 'a+')
    for number in table:
        box_info = " %d %f %f %f %f" % (
            number[0], number[1], number[2], number[3], number[4])
        noise_anno.write(box_info)
        noise_anno.write('\n')
    noise_anno.close()
