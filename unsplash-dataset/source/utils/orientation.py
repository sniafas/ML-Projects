import pandas as pd
from glob import glob
from PIL import Image

# Unsplash raw images path
dataset_path = ''
image_path = glob(dataset_path + '*.jpg')

# Unsplash photo tsv or csv path
photos = pd.read_csv('')
for i in range(len(image_path)):

    im = Image.open(image_path[i])
    im_name = image_path[i].split('/')[-1][:-4]
    if im.width > im.height:
        photos.loc[photos.loc[photos['photo_id'] == im_name].index, 'orientation'] = 1
    else:
        photos.loc[photos.loc[photos['photo_id'] == im_name].index, 'orientation'] = 0
    Image.Image.close(im)

# Write orientation feature in dataset
photos.to_csv("../data/dataset.csv")
