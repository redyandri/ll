from PIL import Image
import os

class Color2Binary(object):
    def convertC2B(self,img_path):
        color=Image.open(img_path)
        #color=color.resize((32,1),Image.ANTIALIAS)
        gray = color.convert("L")
        gray=gray.resize((32,1),Image.ANTIALIAS)
        black_white=gray.point(lambda x: 0 if x<128 else 255, "1")
        black_white.save("binary.png")


c2b=Color2Binary()
c2b.convertC2B("chess.png")
