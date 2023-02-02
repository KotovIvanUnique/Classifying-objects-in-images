import argparse

import matplotlib.pyplot as plt

from mxnet import nd, image
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.imagenet import transform_eval
from matplotlib.pyplot import imshow

from PIL import Image

parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')

opt = parser.parse_args()
opt.model = 'ResNet50_v2'

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
net = get_model(model_name, pretrained=pretrained)

if not pretrained:
    net.load_parameters(opt.saved_params)

print(1)
# Load Images
images = [  "Inception Model/images/human.jpg",
            "Inception Model/images/womanface.jpg",
            "Inception Model/images/cars.jpg",
            "Inception Model/images/panda.jpg",
            "Inception Model/images/snake.jpg",
            "Inception Model/images/anaconda.jpg",
            "Inception Model/images/shark.jpg",
            "Inception Model/images/sword.jpg",
            "Inception Model/images/animals.jpg",
            "Inception Model/images/spider.jpg",
            "Inception Model/images/pandacropped.jpg",
            "Inception Model/images/pandanoise(50.50).jpg",
            "Inception Model/images/pandanoise(75.75).jpg",
            "Inception Model/images/pandanoise(100.100).jpg"]

for im in images :
    #imshow(im)
    #Image.open(im).show()
    print(im)
    img = image.imread(im)
    # Transform
    img = transform_eval(img)
    pred = net(img)
    topK = 10
    ind = nd.topk(pred, k=topK)[0].astype('int')
    for i in range(topK):
        print('\t%.2f%% : %s'%
              (nd.softmax(pred)[0][ind[i]].asscalar(), net.classes[ind[i].asscalar()]))
