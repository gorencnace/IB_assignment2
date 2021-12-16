import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.annotations_path = config['annotations_path']
        self.images_path_plain = config['images_path_plain']
        self.images_path_plain_train = config['images_path_plain_train']
        self.predictions_paths = [config['predictions_path_plain'],
                                  config['predictions_path_denoise'], config['predictions_path_edge'],
                                  config['predictions_path_gamma'], config['predictions_path_gamma2'], config['predictions_path_histo'],
                                  config['predictions_path_kuwahara']]

    def convert_to_pixels(self, size, box):
        w = box[2] * size[0]
        h = box[3] * size[1]
        x = box[0] * size[0] - w / 2
        y = box[1] * size[1] - h / 2
        return [round(x), round(y), round(w), round(h)]

    def get_annotations(self, annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = [float(x) for x in line.split(" ")[1:5]]
                l_arr = self.convert_to_pixels([480, 360], l_arr)
                annot.append(l_arr)
        return annot

    def get_predictions(self, pred_name):
        try:
            with open(pred_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = [float(x) for x in line.split(" ")[1:5]]
                    l_arr = self.convert_to_pixels([480, 360], l_arr)
                    annot.append(l_arr)
        except:
            print("File does not exist!")
            annot = []
        return annot

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path_plain + '/*.png', recursive=True))
        #preprocess = Preprocess()

        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)
            """
            # Apply some preprocessing
            img1 = preprocess.histogram_equlization_rgb(img)# This one makes VJ worse
            print("Histogram done!")
            cv2.imwrite(p[1] + "/" + im_name.split("\\")[-1], img1)
            print("Saving done!")
            img2 = preprocess.gamma_correction(img, 0.6) # best at 0.6
            print("Gamma done!")
            cv2.imwrite(p[3] + "/" + im_name.split("\\")[-1], img2)
            print("Saving done!")
            img3 = preprocess.edge_detection(img)
            print("Edge done!")
            cv2.imwrite(p[5] + "/" + im_name.split("\\")[-1], img3)
            print("Saving done!")
            img4 = preprocess.denoising(img) # 2.41% at 5
            print("Denoising done!")
            cv2.imwrite(p[7] + "/" + im_name.split("\\")[-1], img4)
            print("Saving done!")
            img5 = preprocess.kuwahara(img) # 1.79%
            print("Kuwahara done!")
            cv2.imwrite(p[9] + "/" + im_name.split("\\")[-1], img5)
            print("Saving done!")
            #img6 = preprocess.denoising(img)
            

            img2 = preprocess.gamma_correction(img, 2)  # best at 0.6
            print("Histogram done!")
            cv2.imwrite("data/ears/images/test_gamma_2/" + im_name.split("\\")[-1], img2)
            """

        for pred in self.predictions_paths:
            iou_arr = []
            ap_arr = []
            f1_arr = []
            eval = Evaluation()

            # Change the following detector and/or add your detectors below
            import detectors.cascade_detector.detector as cascade_detector
            # import detectors.your_super_detector.detector as super_detector
            cascade_detector = cascade_detector.Detector()

            """
            p = ["data/ears/train_gamma_2", "data/ears/test_histo","data/ears/train_gamma_2","data/ears/test_gamma",
                         "data/ears/train_gamma_2","data/ears/test_edge", "data/ears/train_gamma_2",
                        "data/ears/test_denoise","data/ears/train_gamma_2","data/ears/test_kuwahara"]
            
            for p in path_list:
                os.mkdir(p)
            """


            for im_name in im_list:

                # Read an image
                img = cv2.imread(im_name)

                # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
                #prediction_list = cascade_detector.detect(img)

                # Read labels_YOLOv3_01_format:
                
                annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
                annot_list = self.get_annotations(annot_name)
                pred_name = os.path.join(pred, Path(os.path.basename(im_name)).stem) + '.txt'
                prediction_list = self.get_predictions(pred_name)
                # Only for detection:
                p, gt = eval.prepare_for_detection(prediction_list, annot_list)

                iou = eval.iou_compute(p, gt)
                iou_arr.append(iou)
                ap = eval.average_precision(p, gt)
                ap_arr.append(ap)
                f1 = eval.f1_score(p, gt)
                f1_arr.append(f1)

            miou = np.average(iou_arr)
            map = np.average(ap_arr)
            mf1 = np.average(f1_arr)
            preprocessing_name = pred.split('\\')[-2]
            print(f"\nPreprocessing: {preprocessing_name}\nAverage IOU: {miou:.2%}\nMean AP: {map:.2%}\nAverage F1: {mf1:.2%}\n")





if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()