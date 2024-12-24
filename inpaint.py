import cv2
import numpy as np
import argparse
import os
# import matplotlib.pyplot as plt # for testing

def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(f"{args.output_dir}/{args.idx}"):
        os.mkdir(f"{args.output_dir}/{args.idx}")
    
    for img_file in os.listdir(f"{args.img_dir}/{args.idx}"):
        img = cv2.imread(f"{args.img_dir}/{args.idx}/{img_file}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 50, 100])
        upper_blue = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        restored_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(f"{args.output_dir}/{args.idx}/{img_file}", restored_img)

    print("Inpainting is done.")
    return 0


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--idx", type=int, required=True, help="video idx")
    argparse.add_argument("--img_dir", type=str, default="datasets/hand/imgs", help="image directory")
    argparse.add_argument("--output_dir", type=str, default="datasets/hand/restored_imgs", help="output directory")
    args = argparse.parse_args()

    main(args)