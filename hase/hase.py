from mnist import MNIST
import random

mndata=MNIST("mnist")
train_image,train_label=mndata.load_training()
print mndata.display(train_image[random.randrange(0,len(train_image))])