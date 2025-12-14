import os

urls = [
    "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
]

os.makedirs("data/food101", exist_ok=True)

for url in urls:
    print("Downloading from:", url)
    os.system(f'curl -L "{url}" --output data/food101/food-101.tar.gz')

print("Download finished. Extracting...")

os.system("tar -xvzf data/food101/food-101.tar.gz -C data/food101")

print("Food-101 dataset is ready.")

