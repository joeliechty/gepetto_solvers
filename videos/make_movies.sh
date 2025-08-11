# Take all PNG frames and turn them into high quality MP4s
ffmpeg -framerate 30 -i frames/tip_force_simulation/%d.png -vf "scale=1280:-2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart tip_force_simulation.mp4
ffmpeg -framerate 30 -i frames/tip_force_inference/%d.png -vf "scale=1280:-2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart tip_force_inference.mp4
ffmpeg -framerate 30 -i frames/distributed_load_simulation/%d.png -vf "scale=1280:-2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart distributed_load_simulation.mp4
ffmpeg -framerate 30 -i frames/distributed_load_inference/%d.png -vf "scale=1280:-2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart distributed_load_inference.mp4
ffmpeg -framerate 30 -i frames/inverse_kinematics/%d.png -vf "scale=1280:-2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart inverse_kinematics.mp4

# Make side-by-side videos of simulation vs. inference for both tip force and dist load cases
ffmpeg -i tip_force_simulation.mp4 -i tip_force_inference.mp4 -filter_complex hstack tip_force_combined.mp4
ffmpeg -i distributed_load_simulation.mp4 -i distributed_load_inference.mp4 -filter_complex hstack dist_load_combined.mp4