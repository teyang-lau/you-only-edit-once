import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime as dt
import uuid


def plot_rgb_hist(img):

    red_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    green_hist = cv2.calcHist([img], [1], None, [256], [0, 255])
    blue_hist = cv2.calcHist([img], [2], None, [256], [0, 255])

    plt.plot(red_hist, color="red")
    plt.plot(green_hist, color="green")
    plt.plot(blue_hist, color="blue")


def saturate(img, factor):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # multiple by a factor to change the saturation
    img[..., 1] = img[..., 1] * factor

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    # https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape/56909036

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= maximum / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    """
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    """

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


def auto_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result


def do_we_need_to_sharpen(image, threshold=50):
    """
    The focus of an image is defined by the variance of Laplacian. We've set the default to 100.
    This is equivalent to taking a photo using a very good camera.

    Reference:
    https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

    """
    var = cv2.Laplacian(image, cv2.CV_64F).var()
    if var <= threshold:
        return True
    else:
        return False


def sharpen_my_image(img, threshold=50):
    """
    This function sharpens the image until the variance of Laplacian is 100.
    """
    kernel_centres = list(range(5, 100))

    for kernel_centre in kernel_centres:
        out_image = None
        sharpen = np.array([[0, -1, 0], [-1, kernel_centre, -1], [0, -1, 0]])

        sharp = cv2.filter2D(img, -1, sharpen)
        if cv2.Laplacian(sharp, cv2.CV_64F).var() > threshold:
            out_image = sharp
            break

    return out_image


# def sharpen_image(input_path, output_path, plot=False):

#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     filename = (os.path.split(input_path)[-1]).split(".")[0]

#     kernel_centres = list(range(5, 100))

#     image = cv2.imread(input_path)

#     if do_we_need_to_sharpen(image) == True:
#         print("Image needs sharpening")
#         for kernel_centre in kernel_centres:
#             out_image = None
#             sharpen = np.array([[0, -1, 0], [-1, kernel_centre, -1], [0, -1, 0]])
#             sharp = cv2.filter2D(image, -1, sharpen)

#             if do_we_need_to_sharpen(sharp) == False:
#                 out_image = sharp
#                 break

#         out = (
#             output_path
#             + "/"
#             + filename
#             + dt.now().strftime("%Y%m%d_%H%M")
#             + "_sharpened.jpg"
#         )

#         cv2.imwrite(out, sharp)

#     else: print('Image does not need sharpening')

#     if plot == True:
#         if do_we_need_to_sharpen(image) == True:
#             fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
#             ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#             ax[0].set(
#                 title=f"Original image: Variance of Laplacian:{cv2.Laplacian(image, cv2.CV_64F).var():.3f}"
#             )
#             ax[0].axis("off")
#             ax[1].imshow(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
#             ax[1].set(
#                 title=f"Sharpened image: Variance of Laplacian:{cv2.Laplacian(out_image, cv2.CV_64F).var():.3f}"
#             )
#             ax[1].axis("off")

#         else:
#             plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#             plt.title("Original image")
#             plt.axis("off")
