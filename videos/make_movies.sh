# Take all PNG frames and turn them into high quality MP4s
make_vid () {
    local input_dir="$1"
    local output_base="$2"

    ffmpeg -framerate 30 -i "$input_dir/%d.png" \
        -vf "scale=1280:-2" \
        -c:v libx264 -preset veryslow -crf 16 \
        -pix_fmt yuv420p -profile:v high -level 4.2 \
        -movflags +faststart "${output_base}.mp4"
}

make_vid frames/kinematics_sim kinematics_sim
make_vid frames/tip_force_sim tip_force_sim
make_vid frames/dist_load_sim dist_load_sim
