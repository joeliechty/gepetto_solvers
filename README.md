May need to do this to resolve dynamic link errors:

echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/gtsam.conf
sudo ldconfig


setup virtual environment with pip etc.
pip install .
run test scripts


To combine GT and estimation simulation videos side by side (requires ffmpeg):
ffmpeg -i simulation.mp4 -i inference.mp4 -filter_complex hstack simulation_and_inference.mp4