# Load video/image/webcam for processing
# Written by: JP Aldama
# Date: 3/13/2020 10:06 pm
# Github: https://github.com/sysad-aldama
# Web: https://www.jeanaldama.info [WIP]
# Email me: quaxiscorp@gmail.com
# Copyright Quaxis Corporation (c) 2020
# Special thanks to: Neha Yadav

import os, cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

model = model_from_json()


