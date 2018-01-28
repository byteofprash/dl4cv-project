#!/bin/bash

python tensorflow/tensorflow/examples/image_retraining/retrain.py --image_dir=data/train --output_graph=experiments/ --output_labels=experiments/ --summaries_dir=experiments/ --model_dir=experiments/ --bottleneck_dir=experiments/bottleneck --random_crop=10 --random_scale=10 --random_brightness=10
