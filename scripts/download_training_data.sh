if [ -z $LNDF_SOURCE_DIR ]; then echo 'Please source "lndf_env.sh" first'
else
# full dataset (~150 GB)
wget -O ndf_mug_data.tar.gz https://www.dropbox.com/s/42owfein4jtobd5/ndf_mug_data.tar.gz?dl=0
wget -O ndf_bottle_data.tar.gz https://www.dropbox.com/s/n90491hu386pg0y/ndf_bottle_data.tar.gz?dl=0
wget -O ndf_bowl_data.tar.gz https://www.dropbox.com/s/q3evi7e39wkhetr/ndf_bowl_data.tar.gz?dl=0
wget -O ndf_occ_data.tar.gz https://www.dropbox.com/s/ok4fb045z7v8cpp/ndf_occ_data.tar.gz?dl=0
mkdir -p $LNDF_SOURCE_DIR/data/training_data
mv ndf_*_data.tar.gz $LNDF_SOURCE_DIR/data/training_data
sleep 5
ls $LNDF_SOURCE_DIR/data/training_data
cd $LNDF_SOURCE_DIR/data/training_data
tar -xzf ndf_mug_data.tar.gz
tar -xzf ndf_bottle_data.tar.gz
tar -xzf ndf_bowl_data.tar.gz
tar -xzf ndf_occ_data.tar.gz
rm ndf_*_data.tar.gz
echo "Training data NDF copied to $LNDF_SOURCE_DIR/data/training_data"
fi
