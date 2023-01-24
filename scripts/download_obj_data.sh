if [ -z $LNDF_SOURCE_DIR ]; then echo 'Please source "lndf_env.sh" first'
else

cd $LNDF_SOURCE_DIR/descriptions
mkdir objects
cd objects

wget -O mug_std_centered_obj_normalized.tar.gz               https://www.dropbox.com/s/re7eynep9zvf0yy/mug_std_centered_obj_normalized.tar.gz?dl=0
wget -O bowl_std_centered_obj_normalized.tar.gz              https://www.dropbox.com/s/kdubql5jdlmyamd/bowl_std_centered_obj_normalized.tar.gz?dl=0
wget -O bottle_std_centered_obj_normalized.tar.gz            https://www.dropbox.com/s/hwudbyp3fjquncq/bottle_std_centered_obj_normalized.tar.gz?dl=0
wget -O bowl_handle_std_centered_obj_normalized.tar.gz       https://www.dropbox.com/s/3iabd1m4u2ikgnw/bowl_handle_std_centered_obj_normalized.tar.gz?dl=0
wget -O bottle_handle_std_centered_obj_normalized.tar.gz     https://www.dropbox.com/s/o628cinpy5rayxx/bottle_handle_std_centered_obj_normalized.tar.gz?dl=0

tar -xzf mug_std_centered_obj_normalized.tar.gz
tar -xzf bowl_std_centered_obj_normalized.tar.gz
tar -xzf bottle_std_centered_obj_normalized.tar.gz
tar -xzf bowl_handle_std_centered_obj_normalized.tar.gz
tar -xzf bottle_handle_std_centered_obj_normalized.tar.gz

rm mug_std_centered_obj_normalized.tar.gz
rm bowl_std_centered_obj_normalized.tar.gz
rm bottle_std_centered_obj_normalized.tar.gz
rm bowl_handle_std_centered_obj_normalized.tar.gz
rm bottle_handle_std_centered_obj_normalized.tar.gz

echo "Object models for NDF copied to $LNDF_SOURCE_DIR/descriptions"

cd $LNDF_SOURCE_DIR
wget -O ndf_other_assets.tar.gz https://www.dropbox.com/s/fopyjjm3fpc3k7i/ndf_other_assets.tar.gz?dl=0
mkdir $LNDF_SOURCE_DIR/assets
mv ndf_other_assets.tar.gz $LNDF_SOURCE_DIR/assets
cd $LNDF_SOURCE_DIR/assets
tar -xzf ndf_other_assets.tar.gz
rm ndf_other_assets.tar.gz
echo "Additional object-related assets copied to $LNDF_SOURCE_DIR/assets"
fi
