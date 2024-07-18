
EPOCH=500
MODEL_DATE=0320_121732 # p7

python test.py --device 0 --config ../models/Noise2Variance/$MODEL_DATE/SIDD_Val/config.json --resume ../models/Noise2Variance/$MODEL_DATE/SIDD_Val/checkpoint-epoch$EPOCH.pth