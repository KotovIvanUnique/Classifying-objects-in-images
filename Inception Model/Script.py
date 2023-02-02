from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import inception

inception
#inception.maybe_download()
model = inception.Inception()

def classify(image_path):
    pred = model.classify(image_path=image_path)
    print(image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)

classify(image_path="images/human.jpg")
classify(image_path="images/womanface.jpg")
classify(image_path="images/cars.jpg")
classify(image_path="images/panda.jpg")
classify(image_path="images/snake.jpg")
classify(image_path="images/anaconda.jpg")
classify(image_path="images/shark.jpg")
classify(image_path="images/sword.jpg")
classify(image_path="images/animals.jpg")
classify(image_path="images/spider.jpg")
classify(image_path="images/pandacropped.jpg")
classify(image_path="images/pandanoise(50.50).jpg")
classify(image_path="images/pandanoise(75.75).jpg")
classify(image_path="images/pandanoise(100.100).jpg")


# print('images/human.jpg'+'\n'+
#         '100.00% : suit'+'\n'+
#         '100.00% : Loafer'+'\n'+
#         '100.00% : Windsor tie'+'\n'+
#         '100.00% : groom'+'\n'+
#         '100.00% : geyser'+'\n'+
#         '100.00% : bow tie'+'\n'+
#         '100.00% : volcano'+'\n'+
#         '100.00% : gown'+'\n'+
#         '100.00% : trench coat'+'\n'+
#         '100.00% : mortarboard')