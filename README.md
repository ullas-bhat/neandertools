# neandertools

`neandertools` provides a simple API for generating image cutouts from Rubin Observatory LSST Butler repositories.

## Quick start

```python
import neandertools as nt

svc = nt.cutouts_from_butler(
    "~/lsst/dp1_subset",
    collections=["LSSTComCam/runs/DRP/DP1/DM-51335"],
)

# dataset_type defaults to "visit_image"
images = svc.cutout(visit=2024110800253, detector=5, x=2036, y=2000, h=201, w=201)

# center by sky coordinates (ra, dec), resolved via image WCS
images = svc.cutout(visit=2024110800253, detector=5, ra=62.1, dec=-31.2, h=201, w=201)

# optional override
images = svc.cutout(
    visit=2024110800253,
    detector=5,
    x=2036,
    y=2000,
    h=201,
    w=201,
    dataset_type="preliminary_visit_image",
)
```

## Notes

- `collections` is required when creating the service.
- `dataset_type` is selected per `cutout(...)` call and defaults to `"visit_image"`.
- `cutout(...)` center must be specified as either (`x`, `y`) or (`ra`, `dec`).
- If center is (`ra`, `dec`), WCS is used to convert to pixel coordinates.
- If `h`/`w` are omitted they default to full image size.
- The sky-coordinate mode (`ra/dec/time`) is wired and needs a `sky_resolver` callback.
