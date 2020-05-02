import numpy as np
from skimage import io

#citim imaginile
images = []
for i in range(9):
    image = np.load("images/car_" + i.__str__() + ".npy")
    images.append(image)

#calcul suma pixelilor a tuturor imaginilor
print("Suma tuturor:", np.sum(images))

#calcul suma pixelilor ptr fiecare imagine
def suma(images):
    smax = 0
    idx = -1
    for i in range(9):
        sum = np.sum(images[i])
        print("imaginea " + i.__str__() + " cu suma " + sum.__str__())
        if sum > smax:
            smax = sum
            idx = i
    print("imaginea cu suma maxima este " + idx.__str__())


#suma(images)


def calc_img_medie(images):
    mean_image = np.zeros((400, 600))
    for img in images:
        mean_image += img
    mean_image /= 9
    io.imshow(mean_image.astype(np.uint8))
    io.show()
    return mean_image

def calc_fgh(images):
    mean_image = calc_img_medie(images)
    dev = np.std(images)
    norm_imgs = []
    for img in images:
        norm_img = (img - mean_image)/dev
        norm_img = norm_img[200:300, 280:400]
        norm_imgs.append(norm_img)
        io.imshow(norm_img.astype(np.uint8))
        io.show()


calc_fgh(images)