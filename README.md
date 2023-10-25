# SLAM

This repository follows the 2013/2014 course on SLAM by Cyrill Stachniss.
Course link: http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/index_en.php
Youtube link: https://www.youtube.com/playlist?list=PLgnQpQtFTOGQrZ4O5QzbIHgl3b1JHimN_

# Running Kalman Filter and its variants

- KF: `python -m src.FILTERS.KF.main`
- EKF: `python -m src.FILTERS.EKF.main`
- UKF: `python -m src.FILTERS.UKF.main`
- IF: `python -m src.FILTERS.IF.main`
- EIF: `python -m src.FILTERS.EIF.main`

# Running SLAM

- EKF_SLAM: `python -m src.SLAM.EKF_SLAM.main`
- SEIF_SLAM: `python -m src.SLAM.SEIF_SLAM.main`

# Output

The output robot trajectories after executing an algorithm will be found in `results`.