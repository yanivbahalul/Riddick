"""Riddick - low-light image and video enhancement.

    from riddick.models import CurveNet
    from riddick.merge import merge_frames

The end-to-end chain (merge -> normalise -> enhance) is riddick/pipeline.py:

    python -m riddick.pipeline --frames burst/ --checkpoint model.pth -o out.png
"""
