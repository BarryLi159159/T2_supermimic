#!/bin/bash
# The in-place deploy clips the Booster_T2 panel offers (config/deploy_config.py SUPERMIMIC_CLIPS ip_*): windows of v4
# training clips (data/npz/t2_twist2), past the capture's T-pose, starting and ending on the calmest arm pose.
# The motions npz are gitignored; run this from the repo root on a new machine.
set -e
SRC=data/npz/t2_twist2
DST=deploy/booster_deploy_supermimic/robots/t2/motions
crop() { python scripts/t2/crop_clip.py "$SRC/$2.npz" "$DST/ip_$1.npz" --start "$3" --end "$4"; }
crop look_around ACCAD_Male2General_c3d_A4-_Look_Around 2.84 13.84
crop swing_arms ACCAD_Male2General_c3d_A3-_Swing_arms 0.72 9.3
crop sway ACCAD_Female1General_c3d_A2_-_Sway_t2 0.72 13.44
crop box_lift ACCAD_Male2General_c3d_A6-_Box_lift 0 10.2
crop bow ACCAD_Male2MartialArtsStances_c3d_D8_-_bow_to_ready 0 4.26
crop salute "Eyes_Japan_Dataset_takiguchi_greeting-04-salute_(head)-takiguchi" 3.8 23.18
crop peace Eyes_Japan_Dataset_kawaguchi_gesture_etc-34-peace-kawaguchi 2.76 6.54
crop shakehands Eyes_Japan_Dataset_takiguchi_greeting-08-shakehands-takiguchi 4.48 21.86
crop stretch MoSh_50027_stretches 0 18.14
crop one_leg Eyes_Japan_Dataset_kawaguchi_pose-17-one_leg-kawaguchi 3.24 21.48
crop boxing HumanEva_S1_Box_3 0.94 13.76
crop karate_jab Eyes_Japan_Dataset_yokoyama_karate-08-jab-yokoyama 2.5 22.18
