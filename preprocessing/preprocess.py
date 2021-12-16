import cv2
import numpy as np
import pykuwahara

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.

    def gamma_correction(self, img, gamma=1.0):
        inv_gamma = 1 / gamma
        # build a table that maps the input pixel values to the output gamma corrected values
        table = np.array([((i / 255) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        # LUT takes the input image and the table and finds the correct mappings for each pixel value
        return cv2.LUT(img, table)

    def edge_detection(self, img):
        img = cv2.Canny(img, 100, 200)
        #cv2.imshow("egdes", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return img

    def denoising(self, img):
        img = cv2.fastNlMeansDenoisingColored(img, None, h=5)
        return img

    def kuwahara(self, img):
        img = pykuwahara.kuwahara(img, method='mean', radius=3)
        return img

    # https://stackoverflow.com/questions/60324296/global-centralize-then-normalize-image
    def global_centering(self, img):
        mean_r = np.sum(img[:,:,0]) / (img.shape[0] * img.shape[1])
        img_red_new = img[:, :, 0] - mean_r
        img_red_new = (img_red_new >= 0) * img_red_new

