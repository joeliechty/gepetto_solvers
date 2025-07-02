May need to do this to resolve dynamic link errors:

echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/gtsam.conf
sudo ldconfig


