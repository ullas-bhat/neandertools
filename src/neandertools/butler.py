"""Butler-backed cutout service."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Union

from astropy.time import Time
from lsst.daf.butler import Butler
from lsst.geom import Box2I, Point2I, SpherePoint, degrees
from lsst.sphgeom import LonLat, UnitVector3d

DataId = dict[str, Any]
SkyResolver = Callable[[float, float, Optional[Union[datetime, str]]], Iterable[DataId]]


class ButlerCutoutService:
    """Generate cutouts from an LSST Butler repository."""

    def __init__(self, butler: Any, sky_resolver: Optional[SkyResolver] = None) -> None:
        self._butler = butler
        self._sky_resolver = sky_resolver
        self._visit_detector_index_cache: dict[str, list[dict[str, Any]]] = {}

    def cutout(
        self,
        ra: Optional[Union[float, Sequence[float]]] = None,
        dec: Optional[Union[float, Sequence[float]]] = None,
        time: Optional[Union[datetime, str]] = None,
        x: Optional[Union[float, Sequence[float]]] = None,
        y: Optional[Union[float, Sequence[float]]] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        dataset_type: str = "visit_image",
        *,
        visit: Optional[Union[int, Sequence[int]]] = None,
        detector: Optional[Union[int, Sequence[int]]] = None,
        limit: Optional[int] = None,
    ) -> list[Any]:
        _validate_request(ra=ra, dec=dec, x=x, y=y, h=h, w=w, visit=visit, detector=detector)
        x_mode = _is_provided(x) or _is_provided(y)
        if x_mode:
            x_values = _as_list(x, "x")
            y_values = _as_list(y, "y")
            assert x_values is not None and y_values is not None
            n_centers = _max_len(x_values, y_values)
            x_values = _broadcast_to(x_values, n_centers, "x")
            y_values = _broadcast_to(y_values, n_centers, "y")
            ra_values = [None] * n_centers
            dec_values = [None] * n_centers
        else:
            ra_values = _as_list(ra, "ra")
            dec_values = _as_list(dec, "dec")
            assert ra_values is not None and dec_values is not None
            n_centers = _max_len(ra_values, dec_values)
            ra_values = _broadcast_to(ra_values, n_centers, "ra")
            dec_values = _broadcast_to(dec_values, n_centers, "dec")
            x_values = [None] * n_centers
            y_values = [None] * n_centers

        if visit is not None:
            visit_values = _as_list(visit, "visit")
            detector_values = _as_list(detector, "detector")
            assert visit_values is not None and detector_values is not None
            n_items = max(len(visit_values), len(detector_values), n_centers)
            visit_values = _broadcast_to(visit_values, n_items, "visit")
            detector_values = _broadcast_to(detector_values, n_items, "detector")
            x_values = _broadcast_to(x_values, n_items, "x")
            y_values = _broadcast_to(y_values, n_items, "y")
            ra_values = _broadcast_to(ra_values, n_items, "ra")
            dec_values = _broadcast_to(dec_values, n_items, "dec")

            out = []
            for v, d, xx, yy, rr, dd in zip(
                visit_values, detector_values, x_values, y_values, ra_values, dec_values
            ):
                image = self._butler.get(dataset_type, dataId={"visit": int(v), "detector": int(d)})
                out.append(self._extract_cutout(image, x=xx, y=yy, ra=rr, dec=dd, h=h, w=w))
            return out

        if self._sky_resolver is None:
            raise NotImplementedError(
                "Sky-position cutouts require a sky_resolver. "
                "Pass one to cutouts_from_butler(..., sky_resolver=...)."
            )

        out = []
        for rr, dd in zip(ra_values, dec_values):
            assert rr is not None and dd is not None
            for data_id in self._sky_resolver(float(rr), float(dd), time):
                image = self._butler.get(dataset_type, dataId=data_id)
                out.append(self._extract_cutout(image, x=None, y=None, ra=float(rr), dec=float(dd), h=h, w=w))
                if limit is not None and len(out) >= limit:
                    return out
        return out

    def find_visit_detector(
        self,
        ra: Union[float, Sequence[float]],
        dec: Union[float, Sequence[float]],
        t: Union[datetime, str, Time, Sequence[datetime], Sequence[str], Sequence[Time]],
        *,
        dataset_type: str = "visit_image",
    ) -> Union[list[tuple[int, int]], list[list[tuple[int, int]]]]:
        """Find (visit, detector) entries containing sky position at exposure time.

        Parameters
        ----------
        ra, dec : float or 1D sequence of float
            Sky coordinates in degrees.
        t : datetime | str | astropy.time.Time | 1D sequence
            Query times. A match requires ``begin <= t < end``.
        dataset_type : str
            Dataset type used to build the visit/detector index.
        """
        ra_values = _as_list(ra, "ra")
        dec_values = _as_list(dec, "dec")
        t_values = _as_list(t, "t")
        assert ra_values is not None and dec_values is not None and t_values is not None

        n = max(len(ra_values), len(dec_values), len(t_values))
        ra_values = _broadcast_to(ra_values, n, "ra")
        dec_values = _broadcast_to(dec_values, n, "dec")
        t_values = _broadcast_to(t_values, n, "t")
        t_values = [_to_astropy_time(v) for v in t_values]

        index = self._get_visit_detector_index(dataset_type)
        all_matches: list[list[tuple[int, int]]] = []

        for ra_i, dec_i, t_i in zip(ra_values, dec_values, t_values):
            point = UnitVector3d(LonLat.fromDegrees(float(ra_i), float(dec_i)))
            matches: list[tuple[int, int]] = []
            for row in index:
                if "timespan" in row:
                    if not row["timespan"].contains(t_i):
                        continue
                else:
                    if not (row["begin"] <= t_i < row["end"]):
                        continue
                if row["region"].contains(point):
                    matches.append((row["visit"], row["detector"]))
            all_matches.append(matches)

        if n == 1:
            return all_matches[0]
        return all_matches

    def _extract_cutout(
        self,
        image: Any,
        x: Optional[float],
        y: Optional[float],
        ra: Optional[float],
        dec: Optional[float],
        h: Optional[int],
        w: Optional[int],
    ) -> Any:
        if not hasattr(image, "getBBox") or not hasattr(image, "Factory"):
            return image

        bbox = image.getBBox()
        if x is not None and y is not None:
            x_center = float(x)
            y_center = float(y)
        else:
            assert ra is not None and dec is not None
            if not hasattr(image, "getWcs") or image.getWcs() is None:
                raise ValueError("Image does not have a WCS; cannot use ra/dec center")
            sky = SpherePoint(ra * degrees, dec * degrees)
            pixel = image.getWcs().skyToPixel(sky)
            x_center = float(pixel.getX())
            y_center = float(pixel.getY())

        w_i = int(w) if w is not None else int(bbox.getMaxX() - bbox.getMinX() + 1)
        h_i = int(h) if h is not None else int(bbox.getMaxY() - bbox.getMinY() + 1)

        x0 = int(round(x_center - (w_i - 1) / 2.0))
        y0 = int(round(y_center - (h_i - 1) / 2.0))
        x1 = x0 + w_i - 1
        y1 = y0 + h_i - 1

        cutout_box = Box2I(Point2I(x0, y0), Point2I(x1, y1))
        try:
            cutout_box.clip(bbox)
        except Exception:
            pass

        return image.Factory(image, cutout_box)

    def _get_visit_detector_index(self, dataset_type: str) -> list[dict[str, Any]]:
        cached = self._visit_detector_index_cache.get(dataset_type)
        if cached is not None:
            return cached

        out: list[dict[str, Any]] = []
        query = self._butler.registry.queryDataIds(["visit", "detector"], datasets=dataset_type).expanded()
        for coord in query:
            if not coord.hasRecords() or coord.region is None:
                continue
            timespan = coord.records["visit"].timespan
            if timespan is None or timespan.begin is None or timespan.end is None:
                continue
            out.append(
                {
                    "visit": int(coord["visit"]),
                    "detector": int(coord["detector"]),
                    "region": coord.region,
                    "timespan": timespan,
                }
            )

        self._visit_detector_index_cache[dataset_type] = out
        return out


def cutouts_from_butler(
    repo: str,
    *,
    collections: Union[str, list[str]],
    butler: Optional[Any] = None,
    sky_resolver: Optional[SkyResolver] = None,
) -> ButlerCutoutService:
    if butler is None:
        butler = Butler(repo, collections=collections)
    return ButlerCutoutService(butler=butler, sky_resolver=sky_resolver)


def _validate_request(
    *,
    ra: Optional[Union[float, Sequence[float]]],
    dec: Optional[Union[float, Sequence[float]]],
    x: Optional[Union[float, Sequence[float]]],
    y: Optional[Union[float, Sequence[float]]],
    h: Optional[int],
    w: Optional[int],
    visit: Optional[Union[int, Sequence[int]]],
    detector: Optional[Union[int, Sequence[int]]],
) -> None:
    xy_mode = _is_provided(x) or _is_provided(y)
    sky_center_mode = _is_provided(ra) or _is_provided(dec)

    if xy_mode and sky_center_mode:
        raise ValueError("Use either (x, y) or (ra, dec) for cutout center, not both")
    if not xy_mode and not sky_center_mode:
        raise ValueError("Provide either (x, y) or (ra, dec) for cutout center")
    if xy_mode and (x is None or y is None):
        raise ValueError("Both x and y must be provided together")
    if sky_center_mode and (ra is None or dec is None):
        raise ValueError("Both ra and dec must be provided together")

    if h is not None and h <= 0:
        raise ValueError("h must be > 0")
    if w is not None and w <= 0:
        raise ValueError("w must be > 0")

    visit_mode = visit is not None or detector is not None

    if visit_mode and (visit is None or detector is None):
        raise ValueError("Both visit and detector must be provided together")

    if not visit_mode and (ra is None or dec is None):
        raise ValueError("Provide either both ra/dec or visit/detector")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_provided(value: Any) -> bool:
    if value is None:
        return False
    if _is_sequence(value):
        return len(value) > 0
    return True


def _as_list(value: Any, name: str) -> Optional[list[Any]]:
    if value is None:
        return None
    if not isinstance(value, (str, bytes, dict)):
        try:
            out = list(value)
        except TypeError:
            out = None
        if out is not None:
            if not out:
                raise ValueError(f"{name} cannot be empty")
            return out
    return [value]


def _max_len(*values: Any) -> int:
    return max(len(v) for v in values if v is not None)


def _broadcast_to(values: list[Any], n: int, name: str) -> list[Any]:
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    raise ValueError(f"{name} must have length 1 or {n}")


def _to_astropy_time(value: Union[datetime, str, Time]) -> Time:
    if isinstance(value, Time):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return Time(value, scale="tai")
        return Time(value)
    return Time(value, scale="tai")
