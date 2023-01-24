if [ -z $LNDF_SOURCE_DIR ]; then echo 'Please source "ndf_env.sh" first'
else
wget -O lndf_bottle_demo.tar.gz https://www.dropbox.com/s/x108xhslaqesmjq/lndf_bottle_demos.tar.gz?dl=0
wget -O lndf_bowl_demo.tar.gz https://www.dropbox.com/s/qola3dowdzlndp2/lndf_bowl_demos.tar.gz?dl=0
wget -O lndf_mug_demo.tar.gz https://www.dropbox.com/s/b9r6wx2ve37zew2/lndf_mug_demos.tar.gz?dl=0
wget -O lndf_mug_handle_demo.tar.gz https://www.dropbox.com/s/9p286eb5wm9hphu/lndf_mug_handle_demos.tar.gz?dl=0

mkdir -p $LNDF_SOURCE_DIR/data/demos
mv lndf_bottle_demo.tar.gz $LNDF_SOURCE_DIR/data/demos
mv lndf_bowl_demo.tar.gz $LNDF_SOURCE_DIR/data/demos
mv lndf_mug_demo.tar.gz $LNDF_SOURCE_DIR/data/demos
mv lndf_mug_handle_demo.tar.gz $LNDF_SOURCE_DIR/data/demos

cd $LNDF_SOURCE_DIR/data/demos
tar -xzf lndf_bottle_demo.tar.gz
tar -xzf lndf_bowl_demo.tar.gz
tar -xzf lndf_mug_demo.tar.gz
tar -xzf lndf_mug_handle_demo.tar.gz

rm lndf_bottle_demo.tar.gz
rm lndf_bowl_demo.tar.gz
rm lndf_mug_demo.tar.gz
rm lndf_mug_handle_demo.tar.gz
echo "Robot demonstrations for NDF copied to $LNDF_SOURCE_DIR/data/demos"
fi
