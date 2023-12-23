echo -e "Downloading T2M evaluators"
cd ./data

gdown "https://drive.google.com/uc?id=1rcYjuawHqq5Z229rIR_dgfTmELNgst0O"

unzip t2m.zip
echo -e "Cleaning\n"
rm t2m.zip

echo -e "Downloading done!"