# Take all PNG frames and turn them into high quality MP4s
make_vid () {
    local input_dir="$1"
    local output_base
    output_base=$(basename "$input_dir")

    ffmpeg -framerate 30 -i "$input_dir/%d.png" \
        -vf "scale=1280:-2" \
        -c:v libx264 -preset veryslow -crf 16 \
        -pix_fmt yuv420p -profile:v high -level 4.2 \
        -movflags +faststart "${output_base}.mp4"
}

make_vid frames/kinematics_sim
make_vid frames/tip_force_prior
make_vid frames/tip_force_nominal
make_vid frames/tip_force_tracking
make_vid frames/dist_load_sim

# Make side by side movie comparing tip force
ffmpeg -i tip_force_nominal.mp4 -i tip_force_tracking.mp4 \
  -filter_complex "[0:v][1:v]hstack=inputs=2" \
  -c:v libx264 -preset veryslow -crf 16 \
  -pix_fmt yuv420p -movflags +faststart \
  tip_force_comparison.mp4
