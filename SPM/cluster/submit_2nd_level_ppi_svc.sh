
# sbatch --array=2 2nd_level_ppi.sh V1_RPE_L urpe S1_R

# sbatch --array=2 2nd_level_ppi.sh V1_RPE_L urpe A1_R
sbatch --array=1 2nd_level_ppi.sh AG_S_L surprise surprise_network
sbatch --array=2 2nd_level_ppi.sh AG_S_L surprise surprise_network
sbatch --array=2 2nd_level_ppi.sh AG_S_L surprise S1_R

sbatch --array=1 2nd_level_ppi.sh AG_S_R surprise surprise_network
sbatch --array=2 2nd_level_ppi.sh AG_S_R surprise S1_R
sbatch --array=2 2nd_level_ppi.sh AG_S_R surprise surprise_network

sbatch --array=1 2nd_level_ppi.sh DLPFC_S_L_surprise surprise surprise_network
sbatch --array=2 2nd_level_ppi.sh DLPFC_S_L surprise surprise_network

# Seed: A1_R:
# More connected during high URPE for audiovisual vs visuotactile? (mask: RPE_network)
# sbatch --array=2 2nd_level_ppi.sh A1_R urpe RPE_network
# # More connected during high surprise for audiovisual vs visuotactile? (mask: surprise_network)
# sbatch --array=2 2nd_level_ppi.sh A1_R surprise surprise_network

# # Seed: S1_R
# # More connected during high URPE for visuotactile vs audiovisual? (mask: RPE_network)
# sbatch --array=2 2nd_level_ppi.sh S1_R urpe RPE_network
# # More connected during high surprise for visuotactile vs audiovisual? (mask: surprise_network)
# sbatch --array=2 2nd_level_ppi.sh S1_R surprise surprise_network

# # Seed: AG_RPE_L
# # More connected during high URPE? (mask: RPE network)
# sbatch --array=1 2nd_level_ppi.sh AG_RPE_L urpe RPE_network
# # More connected during high URPE visuotactile vs audiovisual? (mask: RPE network)
# sbatch --array=2 2nd_level_ppi.sh AG_RPE_L urpe RPE_network
# # More connected during high URPE visuotactile vs audiovisual? (mask: S1_R)
# sbatch --array=2 2nd_level_ppi.sh AG_RPE_L urpe S1_R
# # More connected during high URPE audiovisual vs visuotactile? (mask: RPE network)
# # sbatch --array=2 2nd_level_ppi.sh AG_RPE_L urpe RPE_network
# # More connected during high URPE audiovisual vs visuotactile? (mask: A1_R)
# sbatch --array=2 2nd_level_ppi.sh AG_RPE_L urpe A1_R

# # Seed: V1_RPE_L 
# # More connected during high URPE? (mask: RPE network)
# sbatch --array=1 2nd_level_ppi.sh V1_RPE_L urpe RPE_network
# # More connected during high URPE visuotactile vs audiovisual? (mask: RPE network)
# sbatch --array=2 2nd_level_ppi.sh V1_RPE_L urpe RPE_network
# # More connected during high URPE visuotactile vs audiovisual? (mask: S1_R)
# sbatch --array=2 2nd_level_ppi.sh V1_RPE_L urpe S1_R
# # More connected during high URPE audiovisual vs visuotactile? (mask: RPE network)
# # sbatch --array=2 2nd_level_ppi.sh V1_RPE_L urpe A1_R
# # More connected during high URPE audiovisual vs visuotactile? (mask: A1_R)
# sbatch --array=2 2nd_level_ppi.sh V1_RPE_L urpe A1_R

# # Seed: AG_S_L
# # More connected during high Surprise? (mask: Surprise network)
# sbatch --array=1 2nd_level_ppi.sh AG_S_L surprise Surprise_network
# # More connected during high Surprise for visuotactile vs audiovisual? (mask: Surprise network)
# sbatch --array=2 2nd_level_ppi.sh AG_S_L surprise Surprise_network
# # More connected during high Surprise for visuotactile vs audiovisual? (mask: S1_R)
# sbatch --array=2 2nd_level_ppi.sh AG_S_L surprise S1_R
# # More connected during high Surprise for  audiovisual vs visuotactile? (mask: Surprise network)
# # sbatch --array=2 2nd_level_ppi.sh AG_S_L surprise Surprise_network
# # More connected during high Surprise for  audiovisual vs visuotactile? (mask: A1_R)
# sbatch --array=2 2nd_level_ppi.sh AG_S_L surprise A1_R

# # Seed: AG_S_R
# # More connected during high Surprise? (mask: Surprise network)
# sbatch --array=1 2nd_level_ppi.sh AG_S_R surprise Surprise_network
# # More connected during high Surprise for visuotactile vs audiovisual? (mask: Surprise network)
# sbatch --array=2 2nd_level_ppi.sh AG_S_R surprise Surprise_network
# # More connected during high Surprise for visuotactile vs audiovisual? (mask: S1_R)
# sbatch --array=2 2nd_level_ppi.sh AG_S_R surprise S1_R
# # More connected during high Surprise for  audiovisual vs visuotactile? (mask: Surprise network)
# # sbatch --array=2 2nd_level_ppi.sh AG_S_R surprise Surprise_network
# # More connected during high Surprise for  audiovisual vs visuotactile? (mask: A1_R)
# sbatch --array=2 2nd_level_ppi.sh AG_S_R surprise A1_R

# # Seed: DLPFC_S_L
# # More connected during high Surprise? (mask: Surprise network)
# sbatch --array=1 2nd_level_ppi.sh DLPFC_S_L surprise Surprise_network
# # More connected during high Surprise for visuotactile vs audiovisual? (mask: Surprise network)
# sbatch --array=2 2nd_level_ppi.sh DLPFC_S_L surprise Surprise_network
# # More connected during high Surprise for visuotactile vs audiovisual? (mask: S1_R)
# sbatch --array=2 2nd_level_ppi.sh DLPFC_S_L surprise S1_R
# # More connected during high Surprise for  audiovisual vs visuotactile? (mask: Surprise network)
# # sbatch --array=2 2nd_level_ppi.sh DLPFC_S_L surprise Surprise_network
# # More connected during high Surprise for  audiovisual vs visuotactile? (mask: A1_R)
# sbatch --array=2 2nd_level_ppi.sh DLPFC_S_L surprise A1_R
