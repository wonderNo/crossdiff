echo -e "Downloading pre-trained models"
cd ./data
mkdir ./checkpoints
cd checkpoints

gdown "https://drive.google.com/uc?id=1pKSpIuYES6-ToJPPowwps9LIb_xzTMLE"
gdown "https://drive.google.com/uc?id=13C26tAg2aBU60mwU63DR4dbU_bWvLDWP"
gdown "https://drive.google.com/uc?id=1oMdt1Z8jBulXTqjm8y9or5jBx0IMmzwp"

echo -e "Downloading done!"