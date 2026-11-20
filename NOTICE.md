# Third-party notices

- **Long-WAM / LongLive** (NVlabs, Apache License 2.0): `training/longwam_patches/` and
  `video/stage1_sdpa_fallback.patch` are diffs against https://github.com/NVlabs/LongLive (Long-WAM);
  the two `yam_conveyor.yaml` configs are derived from its RoboTwin 2.0 configs. Apply them to an
  upstream checkout; upstream license terms apply to the patched files.
- **YAMLab** (MIT): the `sim/` task plugs into YAMLab (https://github.com/ARISE-Initiative/yamlab) and
  follows its task structure.
- **HDRI maps** in `sim/assets/hdris/` are 1k maps from Poly Haven (CC0).
- **Model weights** are not included: Long-WAM-YAM and LongLive2.0-Robot-S are downloaded from
  Hugging Face (`Efficient-Large-Model/*`) under their own terms.
