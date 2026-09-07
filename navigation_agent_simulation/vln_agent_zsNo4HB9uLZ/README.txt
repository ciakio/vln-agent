R2R-VLNCE subset for Matterport3D scene zsNo4HB9uLZ

Included episodes:
- trajectory 2027: episode 505, 506, 507
- trajectory 4909: episode 1240, 1241, 1242

This package contains dataset metadata only. Put the existing scene files under:
  /path/to/scene_datasets/mp3d/zsNo4HB9uLZ/

Run from the vln-agent feat/simulation branch:
  python -m simulation.run_agent \
    --dataset /path/to/package/dataset/val_unseen.json.gz \
    --scenes-dir /path/to/scene_datasets \
    --episode-id 505 \
    --record-video

Valid episode IDs: 505, 506, 507, 1240, 1241, 1242.
This is Habitat R2R-VLNCE episode data, not the original R2R train.json format.
