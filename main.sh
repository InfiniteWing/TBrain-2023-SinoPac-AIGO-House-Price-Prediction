#!/bin/sh

pip install -r requirements.txt
sh download_external_data.sh
sh preprocess_external_data.sh
sh feature_engineering.sh
sh model_training_and_prediction.sh