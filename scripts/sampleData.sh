#!/bin/bash

# Helper script to sample and create image dataset
# Usage - ./sampleData.sh <Training_sample_size> <Test_sample_size>

total=$(($1+$2))
mkdir -p data

mkdir -p data/train/Boots-Ankle; find ~/Downloads/ut-zap50k-images/Boots/Ankle -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Boots-Ankle/
mkdir -p data/test/Boots-Ankle; find ~/Downloads/ut-zap50k-images/Boots/Ankle -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Boots-Ankle/

mkdir -p data/train/Boots-Knee; find ~/Downloads/ut-zap50k-images/Boots/Knee\ High -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Boots-Knee/
#find ~/Downloads/ut-zap50k-images/Boots/Over\ the\ Knee -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Boots-Knee/
mkdir -p data/test/Boots-Knee; find ~/Downloads/ut-zap50k-images/Boots/Knee\ High -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Boots-Knee/
#find ~/Downloads/ut-zap50k-images/Boots/Over\ the\ Knee -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Boots-Knee/

mkdir -p data/train/Boots-Midcalf; find ~/Downloads/ut-zap50k-images/Boots/Mid-Calf -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Boots-Midcalf/
mkdir -p data/test/Boots-Midcalf; find ~/Downloads/ut-zap50k-images/Boots/Mid-Calf -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Boots-Midcalf/

mkdir -p data/train/Sandals-Flat; find ~/Downloads/ut-zap50k-images/Sandals/Flat -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Sandals-Flat
mkdir -p data/test/Sandals-Flat; find ~/Downloads/ut-zap50k-images/Sandals/Flat -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Sandals-Flat

mkdir -p data/train/Sandals-Heel; #find ~/Downloads/ut-zap50k-images/Sandals/Heel -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Sandals-Heel
find ~/Downloads/ut-zap50k-images/Shoes/Heels -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Sandals-Heel
mkdir -p data/test/Sandals-Heel; #find ~/Downloads/ut-zap50k-images/Sandals/Heel -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Sandals-Heel
find ~/Downloads/ut-zap50k-images/Shoes/Heels -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Sandals-Heel

mkdir -p data/train/Clogs-Mules; find ~/Downloads/ut-zap50k-images/Shoes/Clogs\ and\ Mules -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Clogs-Mules
mkdir -p data/test/Clogs-Mules; find ~/Downloads/ut-zap50k-images/Shoes/Clogs\ and\ Mules -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Clogs-Mules

mkdir -p data/train/Shoes-Flats; find ~/Downloads/ut-zap50k-images/Shoes/Flats -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Shoes-Flats
mkdir -p data/test/Shoes-Flats; find ~/Downloads/ut-zap50k-images/Shoes/Flats -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Shoes-Flats

mkdir -p data/train/Shoes-Loafers; find ~/Downloads/ut-zap50k-images/Shoes/Loafers -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Shoes-Loafers
mkdir -p data/test/Shoes-Loafers; find ~/Downloads/ut-zap50k-images/Shoes/Loafers -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Shoes-Loafers

mkdir -p data/train/Shoes-Oxfords; find ~/Downloads/ut-zap50k-images/Shoes/Oxfords -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Shoes-Oxfords
mkdir -p data/test/Shoes-Oxfords; find ~/Downloads/ut-zap50k-images/Shoes/Oxfords -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Shoes-Oxfords

mkdir -p data/train/Shoes-Athletic; find ~/Downloads/ut-zap50k-images/Shoes/Sneakers\ and\ Athletic\ Shoes -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Shoes-Athletic
mkdir -p data/test/Shoes-Athletic; find ~/Downloads/ut-zap50k-images/Shoes/Sneakers\ and\ Athletic\ Shoes -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Shoes-Athletic

mkdir -p data/train/Slippers-Flat; find ~/Downloads/ut-zap50k-images/Slippers/Slipper\ Flats -type f -printf '"%p"\n' | head -$1 | xargs cp -t data/train/Slippers-Flat
mkdir -p data/test/Slippers-Flat; find ~/Downloads/ut-zap50k-images/Slippers/Slipper\ Flats -type f -printf '"%p"\n' | head -$total | tail -$2 | xargs cp -t data/test/Slippers-Flat

