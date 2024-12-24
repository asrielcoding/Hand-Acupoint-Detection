import cv2
import os
import json
from ultralytics import  YOLO
import argparse
# from matplotlib import pyplot as plt # for testing

def _video_to_img(video_path, img_dir, idx):
    video_path = f"{video_path}/IMG_{idx}.MOV"
    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_FPS, 60)

    img_dir = img_dir + f"/{idx}"
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    saved_frame_name = 0
    while video_capture.isOpened():
        frame_is_read, frame = video_capture.read()

        if frame_is_read:
            cv2.imwrite(f"{img_dir}/{saved_frame_name}.jpg", frame)
            saved_frame_name += 1

        else:
            print(f"Video to image for {video_path} is done.")
            break

def _get_hands_bbox(img_file, idx):
    save_file = f"datasets/hand/labels/{idx}.jsonl"
    if not os.path.exists(f"datasets/hand/labels"):
        os.mkdir(f"datasets/hand/labels")


    # set model
    model = YOLO('yolov8n.pt')

    results = model(img_file)
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                label = f'Hand: {conf:.2f}'
                j_file = {
                    'image': img_file,
                    'x1': x1.item(),
                    'x2': x2.item(),
                    'y1': y1.item(),
                    'y2': y2.item(),
                    'confidence': conf.item(),
                }

                with open(save_file, 'a') as f:
                    f.write(json.dumps(j_file) + '\n')

def _crop_img(img_file, label_file, idx):
    if not os.path.exists(f"datasets/hand/cropped_imgs"):
        os.mkdir(f"datasets/hand/cropped_imgs")
    if not os.path.exists(f"datasets/hand/cropped_imgs/{idx}"):
        os.mkdir(f"datasets/hand/cropped_imgs/{idx}")

    with open(label_file, 'r') as f:
        for line in f:
            j_file = json.loads(line)

            img = cv2.imread(j_file['image'])
            x1, x2, y1, y2 = j_file['x1'], j_file['x2'], j_file['y1'], j_file['y2']
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]

            cv2.imwrite(f"datasets/hand/cropped_imgs/{idx}/{os.path.basename(j_file['image']).split('.jpg')[0]}_cropped.jpg", crop_img)



def main(args):
    for idx in range(args.st_idx, args.ed_idx):
        label_file = f"datasets/hand/labels/{idx}.jsonl"
        if args.frame:
            _video_to_img(args.video_dir, args.img_dir, idx)
        # dir = datasets/hand/imgs/{idx}
        img_dir = f"{args.img_dir}/{idx}"
        for img_file in os.listdir(img_dir):
            img_file = f"{img_dir}/{img_file}"
            _get_hands_bbox(img_file, idx)
            if args.crop:
                _crop_img(img_file, label_file, idx)

    return 0


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--st_idx', type=int, required=True, help='start index')
    argparse.add_argument('--ed_idx', type=int, required=True, help='end index')
    argparse.add_argument('--video_dir', type=str, default='datasets/hand', help='video directory')
    argparse.add_argument('--img_dir', type=str, default='datasets/hand/imgs', help='output image directory')
    argparse.add_argument('--crop_output', type=str, default='datasets/hand/cropped_imgs', help='output cropped image directory')
    argparse.add_argument('--frame', type=bool, default=False, help='if True, get frame from video.')
    argparse.add_argument('--crop', type=bool, default=False, help='if True, crop image.')
    args = argparse.parse_args()
    main(args)