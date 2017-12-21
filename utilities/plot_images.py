import matplotlib.pyplot as plt
import numpy as np

def in_grid(images, labels=None, rows=1,cols=1):
    if labels is not None:
        labels=np.vstack(labels).astype(np.int)
        label_list=[''.join(str(e) for e in i if e >=0) for i in labels]
    else: label_list=None

    image_list = list(images)

    f, axarr = plt.subplots(nrows=rows, ncols=cols)
    f.subplots_adjust(hspace=0.5)
    f.set_size_inches(8,8)
    axarr=axarr.reshape(rows, cols)
    img_counter=0
    for r in range(rows):
        if img_counter >= len(image_list): break
        for c in range(cols):
            if img_counter >= len(image_list): break
            axarr[r, c].imshow(image_list[img_counter])
            if label_list: axarr[r, c].set_title(label_list[img_counter])
            axarr[r, c].axis('off')
            img_counter +=1
    plt.show()
