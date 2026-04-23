

# import trajectory from ~/git_repos/underactuated_hand/interpolated_trajectory.npz
# unpack time and tendon lengths
# generate synthetic bend sensor data at a higher frequency than the trajecotry frequency (these can be args to pass in)
#   - start bend sensor angles = 0 deg (fully extended)
#   - end bend sensor angles = 0 deg, 45 deg, and 90 deg (linearly interpolate between start and end), this simulates an obstacle being in the way and the finger wrapping around it
#   - frequencies = 200 Hz, 300 Hz, 400 Hz
# run state estimation with the imported tendon trajectories and synthetic bend sensor data, and visualize the results, including the undertainty ellipsoids, will be interesting to see how those change
# plots the tendon length trajectories, bend sensor trajectories, and tip pose trajectories for the different frequencies and bend angles


