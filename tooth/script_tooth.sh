#!/bin/bash

echo "Ejecutando script para entrenar modelo..."

#python3 custom.py train --dataset=/home/lokolte/Proyectos/git/Mask_RCNN/tooth/images --weights=last
#python3 custom.py train --dataset=/home/lokolte/Proyectos/git/Mask_RCNN/tooth/images --weights=coco
python3 custom.py splash --weights=last --image=/home/lokolte/Proyectos/git/Mask_RCNN/tooth/images/train/RG051282_BENTO_THAIS_LUCIA_.jpg
