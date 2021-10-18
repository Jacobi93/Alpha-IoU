import os
import numpy as np

# change noise
# clean_dir = '../coco/labels/val2017/'
# noisy_dir = '../coco/labels/val2017_noise_0.4/'
clean_dir = '../VOC/labels/val_voc/'
noisy_dir = '../VOC/labels/val_voc_noise2/'
IoU_all = 0
for clean_file in os.listdir(clean_dir):
    with open(clean_dir + clean_file, 'r') as f1:
        data_clean = f1.readlines()
    with open(noisy_dir + clean_file, 'r') as f2:
        data_noise = f2.readlines()
    IoU = 0
    if len(data_clean) == len(data_noise):
        for i in range(len(data_clean)):
            # one object
            clean_list = data_clean[i].split(' ')
            noise_list = data_noise[i].split(' ')
            num_clean = []
            num_noise = []
            for j in range(len(clean_list)):
                num_clean.append(float(clean_list[j]))
                num_noise.append(float(noise_list[j+1])) # there is a space before the string
            # c, x, y, w, h
            # Intersection area
            inter = (np.minimum(num_clean[1] + 0.5 * num_clean[3], num_noise[1] + 0.5 * num_noise[3]) - \
                     np.maximum(num_clean[1] - 0.5 * num_clean[3], num_noise[1] - 0.5 * num_noise[3])).clip(0,1) * \
                    (np.minimum(num_clean[2] + 0.5 * num_clean[4], num_noise[2] + 0.5 * num_noise[4]) - \
                     np.maximum(num_clean[2] - 0.5 * num_clean[4], num_noise[2] - 0.5 * num_noise[4])).clip(0,1)

            # Union Area
            union = num_clean[3] * num_clean[4] + num_noise[3] * num_noise[4] - inter

            iou = inter / union
            IoU += iou

        IoU /= len(data_clean)

    else:
        print('Two files do not have same objects!')

    IoU_all += IoU

IoU_all /= len(os.listdir(clean_dir))
print('the IoU over {} images is {}'.format(len(os.listdir(clean_dir)), IoU_all))

