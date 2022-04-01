from asyncore import read
import re
import pdf2image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_img(path):

    img = pdf2image.convert_from_path(path.strip())
    assert len(img) == 1, f"En teoria tot té una sola pàgina, error a: {path} on hi han {len(img)}"

    return np.array(img[0])

class DataAnuncis:
    def __init__(self, filenames) -> None:
        self.files = open(filenames, 'r').readlines()
        pass
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return cv2.cvtColor(read_img(self.files[index]), cv2.COLOR_RGB2GRAY)
    
    def loop(self):
        for file in self.files:
            yield read_img(file)


if __name__ == '__main__':

    files = DataAnuncis('/home/adri/Desktop/cvc/data/tinder-historic/filenames.txt')
    plt.imshow(files[0])
    plt.show()
    for n,file in enumerate(files.loop()):
        print(n, end = '\r')