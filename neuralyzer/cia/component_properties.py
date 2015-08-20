'''
'''

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from scipy import ndimage
import skimage
from skimage import morphology, filters, util, measure


def component_isolation_pipeline(img, return_pipeline=False):

    ppl = []
    img = util.img_as_ubyte(img.copy())

    # 0.
    ppl.append(img)
    # 1. gaussian
    ppl.append(ndimage.gaussian_filter(img, 1))
    # 2. otsu
    ppl.append(img > filters.threshold_otsu(img))
    # 3. erosion
    ppl.append(morphology.erosion(img, morphology.disk(1)))
    # 4. gaussian -> otsu
    ppl.append(ppl[1] > filters.threshold_otsu(ppl[1]))
    # 5. erosion -> otsu
    ppl.append(ppl[3] > filters.threshold_otsu(ppl[3]))
    # 6. otsu -> erosion
    ppl.append(morphology.erosion(ppl[2], morphology.disk(1)))
    # 7. gaussian -> erosion
    ppl.append(morphology.erosion(ppl[1], morphology.disk(1)))
    # 8. gaussian -> otsu -> erosion
    ppl.append(morphology.erosion(ppl[4], morphology.disk(1)))
    # 9. gaussian -> erosion -> otsu
    ppl.append(ppl[7] > filters.threshold_otsu(ppl[7]))
    # 10. gaussian -> otsu -> closing
    ppl.append(morphology.closing(ppl[4], morphology.disk(1)))
    # 11. gaussian -> otsu -> erosion -> closing
    ppl.append(morphology.closing(ppl[8], morphology.disk(1)))
    # 12. gaussian -> otsu -> erosion -> fill_holes
    ppl.append(ndimage.morphology.binary_fill_holes(ppl[8]))
    # 13. erosion -> otsu -> fill_holes
    ppl.append(ndimage.morphology.binary_fill_holes(ppl[5]))
    # 14. erosion -> otsu -> neg background
    tmpimg = ppl[5].copy()
    labels = measure.label(tmpimg, connectivity=2)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    tmpimg[labels != background] = 255
    ppl.append(tmpimg)
    # 15. gaussian -> otsu -> closing -> neg background
    tmpimg = ppl[10].copy()
    labels = measure.label(tmpimg, connectivity=1)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    tmpimg[labels != background] = 255
    ppl.append(tmpimg)

    if return_pipeline: return ppl
    else: return ppl[-1]



def plot_pipeline(ppl, ncols=4, figwidth=16):
    ''' '''
    num_imgs = len(ppl)
    rowheight = figwidth / ncols
    nrows = int(np.ceil(num_imgs/float(ncols)))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figwidth, nrows*rowheight))
    for i, img in enumerate(ppl):
        if nrows <= 1:
            ax[i % ncols].grid(False)
            ax[i % ncols].imshow(img, cmap='gray')
            ax[i % ncols].set_title(str(i))
        else:
            ax[int(i/ncols)][i % ncols].grid(False)
            ax[int(i/ncols)][i % ncols].imshow(img, cmap='gray')
            ax[int(i/ncols)][i % ncols].set_title(str(i))

    fig.tight_layout()

    return fig, ax


def isolate_component(img):
    img = skimage.util.img_as_ubyte(img.copy())
    img = ndimage.gaussian_filter(img, 1)
    img = img > skimage.filters.threshold_otsu(img)
    img = skimage.morphology.closing(img, morphology.disk(1))
    labels = skimage.measure.label(img, connectivity=1)
    label_count = np.bincount(labels.ravel())
    background = np.argmax(label_count)
    img[labels != background] = 255
    return img
