
ffmpeg -framerate 30 -i frames/tip_force_simulation/%d.png -c:v libx264 -pix_fmt yuv420p frames/tip_force_simulation.mp4
ffmpeg -framerate 30 -i frames/tip_force_inference/%d.png -c:v libx264 -pix_fmt yuv420p frames/tip_force_inference.mp4
ffmpeg -framerate 30 -i frames/distributed_load_simulation/%d.png -c:v libx264 -pix_fmt yuv420p frames/distributed_load_simulation.mp4
ffmpeg -framerate 30 -i frames/distributed_load_inference/%d.png -c:v libx264 -pix_fmt yuv420p frames/distributed_load_inference.mp4

ffmpeg -i frames/tip_force_simulation.mp4 -i frames/tip_force_inference.mp4 -filter_complex hstack frames/tip_force_combined.mp4
ffmpeg -i frames/distributed_load_simulation.mp4 -i frames/distributed_load_inference.mp4 -filter_complex hstack frames/dist_load_combined.mp4