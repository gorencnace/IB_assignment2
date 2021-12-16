import os


def convert_to_01(size, box):
    x = box[0] + box[2]/2
    y = box[1] + box[3]/2
    w = box[2]
    h = box[3]
    x = x/size[0]
    w = w/size[0]
    y = y/size[1]
    h = h/size[1]
    return x,y,w,h

def convert_to_pixels(size, box):
    w = box[2] * size[0]
    h = box[3] * size[1]
    x = box[0] * size[0] - w / 2
    y = box[1] * size[1] - h / 2
    return round(x), round(y), round(w), round(h)

def pixels_to_o1():
    path = "./annotations/detection/test_YOLO_format/"
    path_new = "./annotations/detection/test/"
    for filename in os.listdir(path):
        with open(path + filename, "r") as f:
            lines = f.readlines()
            with open(path_new + filename, "w") as fn:
                for line in lines:
                    pos = list(convert_to_01([480, 360], [int(x) for x in line.split(" ")[1:-1]]))
                    fn.write(f"{0} {pos[0]} {pos[1]} {pos[2]} {pos[3]}\n")

def o1_to_pixels():
    path = "./annotations/detection/labels_YOLOv3_01_format/"
    path_new = "./annotations/detection/labels_YOLOv3_normal_format/"
    for filename in os.listdir(path):
        with open(path + filename, "r") as f:
            lines = f.readlines()
            with open(path_new + filename, "w") as fn:
                for line in lines:
                    pos = list(convert_to_pixels([480, 360], [float(x) for x in line.split(" ")[1:]]))
                    fn.write(f"{0} {pos[0]} {pos[1]} {pos[2]} {pos[3]}\n")


o1_to_pixels()
