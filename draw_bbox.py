import random, os, colorsys
import numpy as np
from PIL import Image, ImageDraw


def draw_bboxes(annotation_path, read_image, iname, save_image):
    with open(annotation_path) as f:
        lines = f.readlines()

    # read images
    image = Image.open(read_image + iname)
    draw = ImageDraw.Draw(image)

    for line in lines:
        # Generate colors for drawing bounding boxes.
        # # coco for 80
        # hsv_tuples = [(x / 80, 1., 1.)
        #               for x in range(80)]
        # voc for 20
        hsv_tuples = [(x / 20, 1., 1.)
                      for x in range(20)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        thickness = (image.size[0] + image.size[1]) // 300

        if len(line.split(' ')) == 5: # clean = 5
            x = float(line.split(' ')[1])
            y = float(line.split(' ')[2])
            w = float(line.split(' ')[3])
            h = float(line.split(' ')[4])
            class_num = int(line.split(' ')[0])
        else: # noise = 6
            x = float(line.split(' ')[2])
            y = float(line.split(' ')[3])
            w = float(line.split(' ')[4])
            h = float(line.split(' ')[5])
            class_num = int(line.split(' ')[1])
        # My kingdom for a good redistributable image drawing library.
        x_min = (x - 0.5*w)*image.size[0]
        y_min = (y - 0.5*h)*image.size[1]
        x_max = (x + 0.5*w)*image.size[0]
        y_max = (y + 0.5*h)*image.size[1]
        for i in range(thickness):
            draw.rectangle(
                [x_min + i, y_min + i, x_max - i, y_max - i],
                outline=colors[class_num])
    image.save(save_image + iname)

# # coco, also change the color
# label_dir_clean = '../coco/labels/val2017/'
# label_dir_noise = '../coco/labels/val2017_noise_0.3/'
# read_image = '../coco/images/val2017/'
# save_dir_noise = '../coco/images/noise100/'
# save_dir_clean = '../coco/images/clean100/'

# voc, also change the color
label_dir_clean = '../VOC/labels/val_voc/'
label_dir_noise = '../VOC/labels/val_voc_noise3/'
read_image = '../VOC/images/val_voc/'
save_dir_noise = '../VOC/images/noise100/'
save_dir_clean = '../VOC/images/clean100/'

if not os.path.exists(save_dir_noise):
    os.makedirs(save_dir_noise)
if not os.path.exists(save_dir_clean):
    os.makedirs(save_dir_clean)

label_names = random.sample(os.listdir(label_dir_clean), 100)

# save images with clean annotations
for lname in label_names:
    iname = lname.split('.')[0] + '.jpg'

    # save images with clean annotations
    label1 = label_dir_clean + lname
    draw_bboxes(label1, read_image, iname, save_dir_clean)

    # save images with noisy annotations
    label2 = label_dir_noise + lname
    draw_bboxes(label2, read_image, iname, save_dir_noise)