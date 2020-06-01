import cv2
import torch
import torch.nn as nn
import argparse

from torchvision import transforms
from PIL import Image

import models

torch.backends.cudnn.deterministic = True


def preprocess(frame):
    custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.CenterCrop((120, 120)),
                                           transforms.ToTensor()])
    image = custom_transform(frame)
    device = torch.device('cpu')
    image = image.to(device)
    return image


def main(num_classes, grayscale, model_path, age_offset, device):
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, 30)
    next_bottom_left_corner = (10, 60)
    font_scale = 0.75
    font_color = (255, 255, 255)
    line_type = 2
    model = models.get_model(num_classes, grayscale, model_path, device)
    with torch.no_grad():
        while True:
            ret, frame = cam.read()
            image = Image.fromarray(frame)
            image = preprocess(image)
            image = image.unsqueeze(0)
            _, probas = model(image)
            predict_levels = probas > 0.5
            predicted_label = torch.sum(predict_levels, dim=1)
            cv2.putText(frame, f'Predicted age is: {predicted_label.item() + age_offset}', bottom_left_corner_of_text,
                        font, font_scale, font_color, line_type)
            cv2.putText(frame, 'To quit press Q', next_bottom_left_corner,
                        font, font_scale, font_color, line_type)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    _device = torch.device('cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type',
                        type=str,
                        choices=['UTK', 'AFAD'],
                        default='AFAD'
                        # default='UTK'
                        )
    parser.add_argument('-m', '--model',
                        type=str,
                        default='./models/afad_coral.pt'
                        # default='./models/utk_coral.pt'
                        )
    args = parser.parse_args()
    if args.type == 'AFAD':
        _num_classes = 26
        _age_offset = 15
        _grayscale = False
    elif args.type == 'UTK':
        _num_classes = 40
        _age_offset = 21
        _grayscale = False
    else:
        raise Exception(f'Unknown type {args.type}')
    main(_num_classes, _grayscale, args.model, _age_offset, _device)
