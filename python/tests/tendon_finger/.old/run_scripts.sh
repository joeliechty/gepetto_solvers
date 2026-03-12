mkdir -p log

python sim_kinematics.py > log/sim_kinematics.log &
python sim_tip_force.py > log/sim_tip_force.log &
python sim_dist_load.py > log/sim_dist_load.log &

wait
