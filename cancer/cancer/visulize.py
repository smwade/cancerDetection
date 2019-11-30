from IPython.display import display
from PIL import Image
from matplotlib import pyplot as plt

def show_image(img):
    display(Image.fromarray(img))

def plot_segment(img, segments):
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    for poly in segments:
        plt.fill(poly[:, 0], poly[:, 1], alpha=.3, facecolor='g', edgecolor='black', linewidth=5)

    plt.axis('off')
    plt.show()
