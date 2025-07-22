import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, cast

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from numpy.typing import NDArray
from scipy.ndimage import convolve

from ridge_detector.constants import (
    MAX_ANGLE_DIFFERENCE,
    PIXEL_BOUNDARY,
    cleartab,
    dirtab,
    kernel_c,
    kernel_cc,
    kernel_d,
    kernel_r,
    kernel_rc,
    kernel_rr,
)
from ridge_detector.utils import (
    Crossref,
    Junction,
    Line,
    LineData,
    LinesUtil,
    bresenham,
    closest_point,
    convolve_gauss,
    fix_locations,
    interpolate_gradient_test,
    normalize_to_half_circle,
)


@dataclass
class FilteredData:
    derivatives: NDArray[np.floating]
    lower_thresh: NDArray[np.floating]
    upper_thresh: NDArray[np.floating]
    eigvals: NDArray[np.floating]
    eigvecs: NDArray[np.floating]
    sigma_map: NDArray[np.floating]
    gradx: NDArray[np.floating] = field(init=False)
    grady: NDArray[np.floating] = field(init=False)

    def __post_init__(self):
        self.gradx = self.derivatives[1, ...]
        self.grady = self.derivatives[0, ...]


class LinePoints:
    normx: NDArray[np.floating]
    normy: NDArray[np.floating]
    posx: NDArray[np.floating]
    posy: NDArray[np.floating]
    ismax: NDArray[np.integer]

    def __init__(self, shape: tuple[int, ...]):
        self.ismax = np.zeros(shape, dtype=int)
        self.normx = np.zeros(shape, dtype=float)
        self.normy = np.zeros(shape, dtype=float)
        self.posx = np.zeros(shape, dtype=float)
        self.posy = np.zeros(shape, dtype=float)


class RidgeData:
    """Data structure to hold all ridge detection state and results."""

    # Input data
    image: NDArray[np.uint8]
    gray: NDArray[np.uint8]

    # Line point computation results
    eigval: NDArray[np.floating]
    """For some reason, this is distinct from `FilteredData.eigvals`."""

    # Contour detection results
    contours: list[Line]
    junctions: list[Junction]

    _contour_points: Optional[list[NDArray]] = None
    _width_points: Optional[tuple[list[NDArray], list[NDArray]]] = None

    def __init__(self, image: NDArray[np.floating | np.integer]):
        # Normalize to uint8 if needed. The type checker cannot infer the dtype
        # correctly here, so we use a cast to ensure the type is correct.
        if image.dtype == np.uint8:
            self.image = cast(NDArray[np.uint8], image)
        else:
            self.image = cast(
                NDArray[np.uint8],
                ((image - image.min()) / (image.max() - image.min()) * 255).astype(
                    np.uint8
                ),
            )

        # Convert to grayscale
        self.gray = (
            cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            if self.image.ndim == 3
            else self.image
        ).astype(np.uint8)

        self.eigval = np.zeros(self.shape, dtype=float)
        self.contours = []
        self.junctions = []

    @property
    def shape(self):
        """Return (height, width) of the image."""
        return self.gray.shape[:2]

    @property
    def width(self):
        """Return the width of the image."""
        return self.gray.shape[-1]

    @property
    def height(self):
        """Return the height of the image."""
        return self.gray.shape[-2]

    def export_images(
        self,
        save_dir: Optional[str | Path] = None,
        prefix: str = "",
        make_binary: bool = True,
        draw_junc: bool = False,
        draw_width: bool = True,
    ):
        if save_dir is None:
            save_dir = Path.cwd()
        elif isinstance(save_dir, str):
            save_dir = Path(save_dir)

        contours_image = self.get_image_contours(show_width=False)
        iio.imwrite(save_dir / f"{prefix}_contours.png", contours_image)

        if draw_width:
            contours_image = self.get_image_contours(show_width=True)
            iio.imwrite(save_dir / f"{prefix}_contours_widths.png", contours_image)

        if draw_junc:
            for junc in self.junctions:
                contours_image = cv2.circle(
                    contours_image, (round(junc.x), round(junc.y)), 2, (0, 255, 255), -1
                )
            iio.imwrite(
                save_dir / f"{prefix}_contours_widths_junctions.png", contours_image
            )

        if make_binary:
            binary_contours = self.get_binary_contours()
            iio.imwrite(save_dir / f"{prefix}_binary_contours.png", binary_contours)

            if draw_width:
                binary_width = self.get_binary_widths()
                iio.imwrite(save_dir / f"{prefix}_binary_widths.png", binary_width)

    @property
    def contour_points(self):
        if self._contour_points is not None:
            return self._contour_points
        all_contour_points = []
        height, width = self.shape
        for contour in self.contours:
            contour_points = [
                [
                    LinesUtil.BC(np.round(pt.x), width),
                    LinesUtil.BR(np.round(pt.y), height),
                ]
                for pt in contour
            ]
            all_contour_points.append(np.array(contour_points))
        self._contour_points = all_contour_points
        return self._contour_points

    @property
    def width_points(self) -> tuple[list[NDArray], list[NDArray]]:
        """Return the left and right width points of the contours.

        Returns
        -------
        left : list[NDArray]
        right : list[NDArray]
        """
        if self._width_points is not None:
            return self._width_points
        all_left_edges = []
        all_right_edges = []
        for contour in self.contours:
            left_edge = []
            right_edge = []

            for last_pt, pt in itertools.pairwise(contour):
                nx = np.cos(pt.angle)
                ny = np.sin(pt.angle)
                if last_pt.width_l > 0 and pt.width_l > 0:
                    px_l = np.round(pt.x - pt.width_l * nx)
                    py_l = np.round(pt.y - pt.width_l * ny)
                    left_edge.append([px_l, py_l])
                if last_pt.width_r > 0 and pt.width_r > 0:
                    px_r = np.round(pt.x + pt.width_r * nx)
                    py_r = np.round(pt.y + pt.width_r * ny)
                    right_edge.append([px_r, py_r])
            all_left_edges.append(np.array(left_edge))
            all_right_edges.append(np.array(right_edge))

        self._width_points = (all_left_edges, all_right_edges)
        return self._width_points

    def get_image_contours(
        self,
        show_width=False,
        contour_color=(255, 0, 0),
        width_color=(0, 255, 0),
    ):
        """Plot ridge contours on the image with or without the widths."""
        copied_image = (
            self.image.copy()
            if self.image.ndim > 2
            else np.repeat(self.image[:, :, None], 3, axis=2)
        )
        image = cv2.polylines(copied_image, self.contour_points, False, contour_color)
        if show_width:
            width_left, width_right = self.width_points
            image = cv2.polylines(image, width_right, False, width_color)
            image = cv2.polylines(image, width_left, False, width_color)
        return image

    def get_binary_contours(self):
        """Get a binary image with contours drawn in black."""
        height, width = self.shape
        binary_image = np.ones((height, width), dtype=np.uint8) * 255

        for contour_points in self.contour_points:
            y = np.minimum(contour_points[:, 1], height - 1)
            x = np.minimum(contour_points[:, 0], width - 1)
            binary_image[y, x] = 0
        return binary_image

    def get_binary_widths(self):
        """Get a binary image with ridge widths drawn in black."""
        binary_image = np.ones(self.shape, dtype=np.uint8) * 255
        for width_left, width_right in zip(*self.width_points):
            if width_left.size == 0 or width_right.size == 0:
                continue
            poly_points = np.concatenate((width_left, width_right[::-1, :]), axis=0)
            mask = ski.draw.polygon2mask(self.shape, poly_points[:, [1, 0]])
            binary_image[mask] = 0
        return binary_image


class RidgeDetector:
    data: Optional[RidgeData] = None

    def __init__(
        self,
        line_widths=np.arange(1, 3),
        low_contrast=100,
        high_contrast=200,
        min_len=5,
        max_len=0,
        dark_line=True,
        estimate_width=True,
        extend_line=False,
        correct_pos=False,
    ):
        self.low_contrast = low_contrast
        self.high_contrast = high_contrast
        self.min_len = min_len
        self.max_len = max_len
        self.dark_line = dark_line
        self.estimate_width = estimate_width
        self.extend_line = extend_line
        self.correct_pos = correct_pos

        # Calculate sigmas for multiscale detection
        self.sigmas = np.array([lw / (2 * np.sqrt(3)) + 0.5 for lw in line_widths])

        self.clow = self.low_contrast
        self.chigh = self.high_contrast
        if self.dark_line:
            self.clow = 255 - self.high_contrast
            self.chigh = 255 - self.low_contrast

        # Initialize ridge data container
        self.data = None
        self.mode = LinesUtil.MODE.dark if self.dark_line else LinesUtil.MODE.light

    def apply_filtering(self, data: RidgeData) -> FilteredData:
        if data is None:
            raise ValueError("Ridge data is not initialized.")
        width = data.width
        height = data.height
        num_scales = len(self.sigmas)
        saliency = np.zeros((height, width, num_scales), dtype=float)
        orientation = np.zeros((height, width, 2, num_scales), dtype=float)
        rys = np.zeros((height, width, num_scales), dtype=float)
        rxs = np.zeros((height, width, num_scales), dtype=float)
        ryys = np.zeros((height, width, num_scales), dtype=float)
        rxys = np.zeros((height, width, num_scales), dtype=float)
        rxxs = np.zeros((height, width, num_scales), dtype=float)
        symmetric_image = np.zeros((height, width, 2, 2), dtype=float)

        low_threshs = np.zeros((height, width, num_scales), dtype=float)
        high_threshs = np.zeros((height, width, num_scales), dtype=float)
        sigma_maps = np.zeros((height, width, num_scales), dtype=float)

        # Filtering at different scales
        gray = data.gray.astype(float)
        for scale_idx, sigma in enumerate(self.sigmas):
            ry = convolve_gauss(gray, sigma, LinesUtil.DERIV.R)
            rx = convolve_gauss(gray, sigma, LinesUtil.DERIV.C)
            ryy = convolve_gauss(gray, sigma, LinesUtil.DERIV.RR)
            rxy = convolve_gauss(gray, sigma, LinesUtil.DERIV.RC)
            rxx = convolve_gauss(gray, sigma, LinesUtil.DERIV.CC)

            symmetric_image[..., 0, 0] = ryy
            symmetric_image[..., 0, 1] = rxy
            symmetric_image[..., 1, 0] = rxy
            symmetric_image[..., 1, 1] = rxx
            eigvals, eigvecs = np.linalg.eigh(symmetric_image)

            # Maximum absolute eigen as the saliency of lines
            idx = np.absolute(eigvals).argsort()[..., ::-1]
            eigvals_tmp = np.take_along_axis(eigvals, idx, axis=-1)
            eigvecs_tmp = np.take_along_axis(eigvecs, idx[:, :, None, :], axis=-1)

            saliency[:, :, scale_idx] = sigma**2.0 * eigvals_tmp[:, :, 0]
            orientation[:, :, :, scale_idx] = eigvecs_tmp[:, :, :, 0]

            # Store intermediate results
            rys[..., scale_idx] = ry
            rxs[..., scale_idx] = rx
            ryys[..., scale_idx] = ryy
            rxys[..., scale_idx] = rxy
            rxxs[..., scale_idx] = rxx

            # Calculate thresholds for each scale using gamma-normalized measurement (gamma is to 2.0)
            line_width = 2 * np.sqrt(3) * (sigma - 0.5)
            low_thresh = (
                0.17
                * sigma**2.0
                * np.floor(
                    self.clow
                    * line_width
                    / (np.sqrt(2 * np.pi) * sigma**3)
                    * np.exp(-(line_width**2) / (8 * sigma**2))
                )
            )
            high_thresh = (
                0.17
                * sigma**2.0
                * np.floor(
                    self.chigh
                    * line_width
                    / (np.sqrt(2 * np.pi) * sigma**3)
                    * np.exp(-(line_width**2) / (8 * sigma**2))
                )
            )
            low_threshs[..., scale_idx] = low_thresh
            high_threshs[..., scale_idx] = high_thresh
            sigma_maps[..., scale_idx] = sigma

        # Get the scale index of the maximum saliency and the corresponding derivatives and thresholds
        global_max_idx = saliency.argsort()[..., -1]
        derivatives = np.squeeze(
            np.array(
                [
                    np.take_along_axis(rys, global_max_idx[:, :, None], axis=-1),
                    np.take_along_axis(rxs, global_max_idx[:, :, None], axis=-1),
                    np.take_along_axis(ryys, global_max_idx[:, :, None], axis=-1),
                    np.take_along_axis(rxys, global_max_idx[:, :, None], axis=-1),
                    np.take_along_axis(rxxs, global_max_idx[:, :, None], axis=-1),
                ]
            )
        )
        return FilteredData(
            lower_thresh=np.squeeze(
                np.take_along_axis(low_threshs, global_max_idx[:, :, None], axis=-1)
            ),
            upper_thresh=np.squeeze(
                np.take_along_axis(high_threshs, global_max_idx[:, :, None], axis=-1)
            ),
            sigma_map=np.squeeze(
                np.take_along_axis(sigma_maps, global_max_idx[:, :, None], axis=-1)
            ),
            derivatives=derivatives,
            eigvals=np.take_along_axis(saliency, global_max_idx[:, :, None], axis=-1),
            eigvecs=np.take_along_axis(
                orientation, global_max_idx[:, :, None, None], axis=-1
            ),
        )

    def compute_line_points(
        self, ridge_data: RidgeData, filtered_data: FilteredData
    ) -> LinePoints:
        ry = filtered_data.grady
        rx = filtered_data.gradx
        ryy = filtered_data.derivatives[2, ...]
        rxy = filtered_data.derivatives[3, ...]
        rxx = filtered_data.derivatives[4, ...]

        val = filtered_data.eigvals[:, :, 0] * self.mode.value
        val_mask = val > 0.0
        ridge_data.eigval[val_mask] = val[val_mask]

        # Equations (22) and (23) in
        # Steger, C. (1998). An unbiased detector of curvilinear structures. (DOI: 10.1109/34.659930)
        nx = filtered_data.eigvecs[..., 1, 0]
        ny = filtered_data.eigvecs[..., 0, 0]
        numerator = (ry * ny) + (rx * nx)
        denominator = (ryy * ny**2) + (2.0 * rxy * nx * ny) + (rxx * nx**2)

        # The minus sign in Eq. (23) is ignored to follow the logic of detecting black ridges in white background
        t = numerator / (denominator + np.finfo(float).eps)
        py = t * ny
        px = t * nx

        bnd_mask = (abs(py) <= PIXEL_BOUNDARY) & (abs(px) <= PIXEL_BOUNDARY)
        base_mask = val_mask & bnd_mask
        upper_mask = base_mask & (val >= filtered_data.upper_thresh)
        lower_mask = (
            base_mask
            & (val >= filtered_data.lower_thresh)
            & (val < filtered_data.upper_thresh)
        )

        line_data = LinePoints(ridge_data.shape)
        line_data.ismax[upper_mask] = 2
        line_data.ismax[lower_mask] = 1

        line_data.normy[base_mask] = ny[base_mask]
        line_data.normx[base_mask] = nx[base_mask]
        Y, X = np.mgrid[: ridge_data.height, : ridge_data.width]
        line_data.posy[base_mask] = Y[base_mask] + py[base_mask]
        line_data.posx[base_mask] = X[base_mask] + px[base_mask]
        return line_data

    def extend_lines(
        self,
        ridge_data: RidgeData,
        label: NDArray[np.integer],
        filtered_data: FilteredData,
    ):
        height, width = label.shape[:2]
        s = self.mode.value
        length = 2.5 * filtered_data.sigma_map
        max_line = np.ceil(length * 1.2).astype(int)
        for idx_cont, contour in enumerate(ridge_data.contours):
            num_pnt = contour.num
            if (
                len(contour) == 1
                or contour.get_contour_class() == LinesUtil.ContourClass.cont_closed
            ):
                continue

            # Assign initial values for m and j so they aren't unbound
            other_contour_idx = 0
            other_contour = None
            j = 0
            end_angle = 0
            end_resp = 0
            # Check both ends of the line (it==-1: start, it==1: end).
            for it in [-1, 1]:
                trow = contour.row
                tcol = contour.col
                tangle = contour.angle
                tresp = contour.response
                if it == -1:
                    # Start point of the line.
                    if contour.get_contour_class() in [
                        LinesUtil.ContourClass.cont_start_junc,
                        LinesUtil.ContourClass.cont_both_junc,
                    ]:
                        continue
                    dy, dx = trow[1] - trow[0], tcol[1] - tcol[0]
                    alpha = tangle[0]
                    ny, nx = np.sin(alpha), np.cos(alpha)
                    if ny * dx - nx * dy < 0:
                        # Turn the normal by +90 degrees.
                        my, mx = -nx, ny
                    else:
                        # Turn the normal by -90 degrees.
                        my, mx = nx, -ny
                    py, px = trow[0], tcol[0]
                    response = tresp[0]
                else:
                    # End point of the line.
                    if contour.get_contour_class() in [
                        LinesUtil.ContourClass.cont_end_junc,
                        LinesUtil.ContourClass.cont_both_junc,
                    ]:
                        continue
                    dy = trow[num_pnt - 1] - trow[num_pnt - 2]
                    dx = tcol[num_pnt - 1] - tcol[num_pnt - 2]
                    alpha = tangle[num_pnt - 1]
                    ny, nx = np.sin(alpha), np.cos(alpha)
                    if ny * dx - nx * dy < 0:
                        # Turn the normal by -90 degrees.
                        my, mx = nx, -ny
                    else:
                        # Turn the normal by +90 degrees.
                        my, mx = -nx, ny
                    py, px = trow[num_pnt - 1], tcol[num_pnt - 1]
                    response = tresp[num_pnt - 1]

                # Determine the current pixel and calculate the pixels on the search line.
                y, x = int(py + 0.5), int(px + 0.5)
                dy, dx = py - y, px - x
                line_length = max_line[LinesUtil.BR(y, height), LinesUtil.BC(x, width)]
                line = bresenham(my, mx, line_length, dy, dx)
                num_line = line.shape[0]
                exty = np.zeros(num_line, dtype=int)
                extx = np.zeros(num_line, dtype=int)

                # Now determine whether we can go only uphill (bright lines)
                # or downhill (dark lines) until we hit another line.
                num_add = 0
                add_ext = False
                for k in range(num_line):
                    nexty, nextx = y + line[k, 0], x + line[k, 1]
                    nextpy, nextpx, t = closest_point(py, px, my, mx, nexty, nextx)

                    # Ignore points before or less than half a pixel away from the true end point of the line.
                    if t <= 0.5:
                        continue
                    # Stop if the gradient can't be interpolated any more or if the next point lies outside the image.
                    if (
                        nextpx < 0
                        or nextpy < 0
                        or nextpy >= height - 1
                        or nextpx >= width - 1
                        or nextx < 0
                        or nexty < 0
                        or nexty >= height
                        or nextx >= width
                    ):
                        break
                    gy, gx = interpolate_gradient_test(
                        filtered_data.grady, filtered_data.gradx, nextpy, nextpx
                    )

                    # Stop if we can't go uphill anymore.
                    # This is determined by the dot product of the line direction and the gradient.
                    # If it is smaller than 0 we go downhill (reverse for dark lines).
                    if s * (mx * gx + my * gy) < 0 and label[nexty, nextx] == 0:
                        break
                    # Have we hit another line?
                    if label[nexty, nextx] > 0:
                        other_contour_idx = label[nexty, nextx] - 1
                        other_contour = ridge_data.contours[other_contour_idx]
                        # Search for the junction point on the other line.
                        dist = np.sqrt(
                            (nextpy - other_contour.row) ** 2
                            + (nextpx - other_contour.col) ** 2
                        )
                        j = np.argmin(dist)

                        exty[num_add] = other_contour.row[j]
                        extx[num_add] = other_contour.col[j]
                        end_resp = other_contour.response[j]
                        end_angle = other_contour.angle[j]
                        beta = end_angle
                        if beta >= np.pi:
                            beta -= np.pi
                        diff1 = abs(beta - alpha)
                        if diff1 >= np.pi:
                            diff1 = 2.0 * np.pi - diff1
                        diff2 = abs(beta + np.pi - alpha)
                        if diff2 >= np.pi:
                            diff2 = 2.0 * np.pi - diff2
                        if diff1 < diff2:
                            end_angle = beta
                        else:
                            end_angle = beta + np.pi
                        num_add += 1
                        add_ext = True
                        break
                    else:
                        exty[num_add], extx[num_add] = nextpy, nextpx
                        num_add += 1

                if add_ext:
                    # Make room for the new points.
                    num_pnt += num_add
                    new_row = np.zeros(num_pnt, dtype=float)
                    new_col = np.zeros(num_pnt, dtype=float)
                    new_angle = np.zeros(num_pnt, dtype=float)
                    new_resp = np.zeros(num_pnt, dtype=float)

                    contour.row = new_row
                    contour.col = new_col
                    contour.angle = new_angle
                    contour.response = new_resp
                    if it == -1:
                        contour.row[num_add:] = trow
                        contour.row[:num_add] = exty[:num_add][::-1]
                        contour.col[num_add:] = tcol
                        contour.col[:num_add] = extx[:num_add][::-1]
                        contour.angle[num_add:] = tangle
                        contour.angle[:num_add] = float(alpha)
                        contour.response[num_add:] = tresp
                        contour.response[:num_add] = float(response)
                        contour.angle[0] = end_angle
                        contour.response[0] = end_resp
                        # Adapt indices of the previously found junctions.
                        for k in range(len(ridge_data.junctions)):
                            if ridge_data.junctions[k].cont1 == idx_cont:
                                ridge_data.junctions[k].pos += num_add
                    else:
                        # Insert points at the end of the line.
                        contour.row[: num_pnt - num_add] = trow
                        contour.row[num_pnt - num_add :] = exty[:num_add]
                        contour.col[: num_pnt - num_add] = tcol
                        contour.col[num_pnt - num_add :] = extx[:num_add]
                        contour.angle[: num_pnt - num_add] = tangle
                        contour.angle[num_pnt - num_add :] = float(alpha)
                        contour.response[: num_pnt - num_add] = tresp
                        contour.response[num_pnt - num_add :] = float(response)
                        contour.angle[-1] = end_angle
                        contour.response[-1] = end_resp

                    # Add the junction point only if it is not one of the other line's endpoints.
                    if other_contour is not None and 0 < j < other_contour.num - 1:
                        if it == -1:
                            if (
                                contour.get_contour_class()
                                == LinesUtil.ContourClass.cont_end_junc
                            ):
                                contour.set_contour_class(
                                    LinesUtil.ContourClass.cont_both_junc
                                )
                            else:
                                contour.set_contour_class(
                                    LinesUtil.ContourClass.cont_start_junc
                                )
                        else:
                            if (
                                contour.get_contour_class()
                                == LinesUtil.ContourClass.cont_start_junc
                            ):
                                contour.set_contour_class(
                                    LinesUtil.ContourClass.cont_both_junc
                                )
                            else:
                                contour.set_contour_class(
                                    LinesUtil.ContourClass.cont_end_junc
                                )

                        if it == -1:
                            contour_idx = 0
                        else:
                            contour_idx = num_pnt - 1
                        ridge_data.junctions.append(
                            Junction(
                                int(other_contour_idx),
                                int(idx_cont),
                                int(j),
                                contour.row[contour_idx],
                                contour.col[contour_idx],
                            )
                        )
            return ridge_data.junctions

    def compute_contours(
        self,
        ridge_data: RidgeData,
        filtered_data: FilteredData,
        line_points: LinePoints,
    ):
        width = ridge_data.width
        height = ridge_data.height
        label = np.zeros((height, width), dtype=int)
        indx = np.zeros((height, width), dtype=int)

        cross: list[Crossref] = []
        for r_idx, c_idx in itertools.product(range(height), range(width)):
            if line_points.ismax[r_idx, c_idx] >= 2:
                cross.append(
                    Crossref(r_idx, c_idx, ridge_data.eigval[r_idx, c_idx], False)
                )
        area = len(cross)

        response_2d = ridge_data.eigval.reshape(height, width)
        resp_dr = convolve(response_2d, kernel_r, mode="mirror")
        resp_dc = convolve(response_2d, kernel_c, mode="mirror")
        resp_dd = convolve(response_2d, kernel_d, mode="mirror")
        resp_drr = convolve(response_2d, kernel_rr, mode="mirror")
        resp_drc = convolve(response_2d, kernel_rc, mode="mirror")
        resp_dcc = convolve(response_2d, kernel_cc, mode="mirror")

        # Sorting cross list in ascending order by value
        cross.sort()

        # Updating indx based on the sorted cross list
        for ci, cref in enumerate(cross):
            indx[cref.y, cref.x] = ci + 1

        indx_max = 0
        while True:
            cls = LinesUtil.ContourClass.cont_no_junc
            while indx_max < area and cross[indx_max].done:
                indx_max += 1

            if indx_max == area:
                break

            max_val = cross[indx_max].value
            maxy, maxx = cross[indx_max].y, cross[indx_max].x
            if max_val == 0.0:
                break

            # Initialize line data
            line_data = LineData()

            # Add starting point to the line.
            label[maxy, maxx] = len(ridge_data.contours) + 1
            if indx[maxy, maxx] != 0:
                cross[indx[maxy, maxx] - 1].done = True

            # Select line direction
            nx = -line_points.normx[maxy, maxx]
            ny = line_points.normy[maxy, maxx]
            alpha = normalize_to_half_circle(np.arctan2(ny, nx))
            octant = int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4

            """ * Select normal to the line. The normal points to the right of the line as the
                * line is traversed from 0 to num-1. Since the points are sorted in reverse
                * order before the second iteration, the first beta actually has to point to
                * the left of the line!
            """

            beta = alpha + np.pi / 2.0
            if beta >= 2.0 * np.pi:
                beta -= 2.0 * np.pi
            yy = line_points.posy[maxy, maxx] - maxy
            xx = line_points.posx[maxy, maxx] - maxx
            interpolated_response = (
                resp_dd[maxy, maxx]
                + yy * resp_dr[maxy, maxx]
                + xx * resp_dc[maxy, maxx]
                + yy**2 * resp_drr[maxy, maxx]
                + xx * yy * resp_drc[maxy, maxx]
                + xx**2 * resp_dcc[maxy, maxx]
            )
            line_data.append(
                row=maxy, col=maxx, angle=beta, response=interpolated_response
            )

            # Mark double responses as processed.
            for ni in range(2):
                nexty = maxy + cleartab[octant][ni][0]
                nextx = maxx + cleartab[octant][ni][1]
                if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                    continue
                if line_points.ismax[nexty, nextx] > 0:
                    nx = -line_points.normx[nexty, nextx]
                    ny = line_points.normy[nexty, nextx]
                    nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                    diff = abs(alpha - nextalpha)
                    if diff >= np.pi / 2.0:
                        diff = np.pi - diff
                    if diff < MAX_ANGLE_DIFFERENCE:
                        label[nexty, nextx] = len(ridge_data.contours) + 1
                        if indx[nexty, nextx] != 0:
                            cross[indx[nexty, nextx] - 1].done = True

            for it in range(1, 3):
                y, x = maxy, maxx
                ny, nx = line_points.normy[y, x], -line_points.normx[y, x]

                alpha = normalize_to_half_circle(np.arctan2(ny, nx))
                last_octant = (
                    int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4
                    if it == 1
                    else int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4 + 4
                )
                last_beta = alpha + np.pi / 2.0
                if last_beta >= 2.0 * np.pi:
                    last_beta -= 2.0 * np.pi

                if it == 2:
                    # Sort the points found in the first iteration in reverse.
                    line_data.reverse()

                while True:
                    ny, nx = line_points.normy[y, x], -line_points.normx[y, x]
                    py, px = line_points.posy[y, x], line_points.posx[y, x]

                    # Orient line direction with respect to the last line direction
                    alpha = normalize_to_half_circle(np.arctan2(ny, nx))
                    octant = int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4

                    if octant == 0 and 3 <= last_octant <= 5:
                        octant = 4
                    elif octant == 1 and 4 <= last_octant <= 6:
                        octant = 5
                    elif octant == 2 and 4 <= last_octant <= 7:
                        octant = 6
                    elif octant == 3 and (last_octant == 0 or last_octant >= 6):
                        octant = 7
                    last_octant = octant

                    # Determine appropriate neighbor
                    nextismax = False
                    nexti = 1
                    mindiff = float("inf")
                    for ti in range(3):
                        nexty, nextx = (
                            y + dirtab[octant][ti][0],
                            x + dirtab[octant][ti][1],
                        )
                        if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                            continue
                        if line_points.ismax[nexty, nextx] == 0:
                            continue
                        nextpy, nextpx = (
                            line_points.posy[nexty, nextx],
                            line_points.posx[nexty, nextx],
                        )
                        dy = nextpy - py
                        dx = nextpx - px
                        dist = np.sqrt(dx**2 + dy**2)
                        ny = line_points.normy[nexty, nextx]
                        nx = -line_points.normx[nexty, nextx]
                        nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                        diff = abs(alpha - nextalpha)
                        if diff >= np.pi / 2.0:
                            diff = np.pi - diff
                        diff = dist + diff
                        if diff < mindiff:
                            mindiff = diff
                            nexti = ti
                        if not (line_points.ismax[nexty, nextx] == 0):
                            nextismax = True

                    # Mark double responses as processed
                    for ni in range(2):
                        nexty = y + cleartab[octant][ni][0]
                        nextx = x + cleartab[octant][ni][1]
                        if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                            continue
                        if line_points.ismax[nexty, nextx] > 0:
                            ny = line_points.normy[nexty, nextx]
                            nx = -line_points.normx[nexty, nextx]
                            nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                            diff = abs(alpha - nextalpha)
                            if diff >= np.pi / 2.0:
                                diff = np.pi - diff
                            if diff < MAX_ANGLE_DIFFERENCE:
                                label[nexty, nextx] = len(ridge_data.contours) + 1
                                if not (indx[nexty, nextx] == 0):
                                    cross[indx[nexty, nextx] - 1].done = True

                    # Have we found the end of the line?
                    if not nextismax:
                        break  # Exit the loop if the end of the line is found

                    # Add the neighbor to the line if not at the end
                    y += dirtab[octant][nexti][0]
                    x += dirtab[octant][nexti][1]

                    # Orient normal to the line direction with respect to the last normal
                    ny = line_points.normy[y, x]
                    nx = line_points.normx[y, x]

                    beta = normalize_to_half_circle(np.arctan2(ny, nx))
                    diff1 = np.minimum(
                        abs(beta - last_beta), 2.0 * np.pi - abs(beta - last_beta)
                    )
                    # Normalize alternative beta
                    alt_beta = (beta + np.pi) % (2.0 * np.pi)
                    diff2 = np.minimum(
                        abs(alt_beta - last_beta),
                        2.0 * np.pi - abs(alt_beta - last_beta),
                    )
                    # Choose the angle with the smallest difference and update
                    chosen_beta = beta if diff1 < diff2 else alt_beta
                    last_beta = chosen_beta

                    yy = line_points.posy[y, x] - maxy
                    xx = line_points.posx[y, x] - maxx
                    interpolated_response = (
                        resp_dd[y, x]
                        + yy * resp_dr[y, x]
                        + xx * resp_dc[y, x]
                        + yy**2 * resp_drr[y, x]
                        + xx * yy * resp_drc[y, x]
                        + xx**2 * resp_dcc[y, x]
                    )
                    line_data.append(
                        row=line_points.posy[y, x],
                        col=line_points.posx[y, x],
                        angle=chosen_beta,
                        response=interpolated_response,
                    )

                    # If the appropriate neighbor is already processed a junction point is found
                    if label[y, x] > 0:
                        k = label[y, x] - 1
                        if k == len(ridge_data.contours):
                            # Line intersects itself
                            for j in range(len(line_data) - 1):
                                if not (
                                    line_data.row[j] == line_points.posy[y, x]
                                    and line_data.col[j] == line_points.posx[y, x]
                                ):
                                    continue
                                if j == 0:
                                    # Contour is closed
                                    cls = LinesUtil.ContourClass.cont_closed
                                    line_data.reverse()
                                    it = 2
                                else:
                                    # Determine contour class
                                    if it == 2:
                                        if (
                                            cls
                                            == LinesUtil.ContourClass.cont_start_junc
                                        ):
                                            cls = LinesUtil.ContourClass.cont_both_junc
                                        else:
                                            cls = LinesUtil.ContourClass.cont_end_junc
                                        # Index j is correct
                                        pos = j
                                    else:
                                        cls = LinesUtil.ContourClass.cont_start_junc
                                        # Index num_pnt-1-j is correct since the line will be sorted in reverse
                                        pos = len(line_data) - 1 - j
                                    ridge_data.junctions.append(
                                        Junction(
                                            len(ridge_data.contours),
                                            len(ridge_data.contours),
                                            pos,
                                            float(line_points.posy[y, x]),
                                            float(line_points.posx[y, x]),
                                        )
                                    )
                                break
                            j = -1
                        else:
                            for j in range(ridge_data.contours[k].num):
                                if (
                                    ridge_data.contours[k].row[j]
                                    == line_points.posy[y, x]
                                    and ridge_data.contours[k].col[j]
                                    == line_points.posx[y, x]
                                ):
                                    break
                            else:
                                j = -1
                            if j == ridge_data.contours[k].num:
                                # No point found on the other line, a double response occurred
                                dist = np.sqrt(
                                    (
                                        line_points.posy[y, x]
                                        - ridge_data.contours[k].row
                                    )
                                    ** 2
                                    + (
                                        line_points.posx[y, x]
                                        - ridge_data.contours[k].col
                                    )
                                    ** 2
                                )
                                j = np.argmin(dist)
                                beta = ridge_data.contours[k].angle[j]
                                if beta >= np.pi:
                                    beta -= np.pi
                                diff1 = abs(beta - last_beta)
                                if diff1 >= np.pi:
                                    diff1 = 2.0 * np.pi - diff1
                                diff2 = abs(beta + np.pi - last_beta)
                                if diff2 >= np.pi:
                                    diff2 = 2.0 * np.pi - diff2
                                line_data.append(
                                    row=ridge_data.contours[k].row[j],
                                    col=ridge_data.contours[k].col[j],
                                    angle=beta if diff1 < diff2 else beta + np.pi,
                                    response=ridge_data.contours[k].response[j],
                                )
                        if 0 < j < ridge_data.contours[k].num - 1:
                            # Determine contour class
                            if it == 1:
                                cls = LinesUtil.ContourClass.cont_start_junc
                            elif cls == LinesUtil.ContourClass.cont_start_junc:
                                cls = LinesUtil.ContourClass.cont_both_junc
                            else:
                                cls = LinesUtil.ContourClass.cont_end_junc

                            # Add the new junction
                            ridge_data.junctions.append(
                                Junction(
                                    int(k),
                                    len(ridge_data.contours),
                                    int(j),
                                    line_data.row[-1],
                                    line_data.col[-1],
                                )
                            )
                        break

                    label[y, x] = len(ridge_data.contours) + 1
                    if indx[y, x] != 0:
                        cross[indx[y, x] - 1].done = True

            if len(line_data) > 1:
                ridge_data.contours.append(line_data.to_line(contour_class=cls))
            else:
                # Delete the point from the label image; using maxx and maxy as coordinates in the label image
                for i, j in itertools.product(range(-1, 2), repeat=2):
                    if (
                        label[
                            LinesUtil.BR(maxy + i, height),
                            LinesUtil.BC(maxx + j, width),
                        ]
                        == len(ridge_data.contours) + 1
                    ):
                        label[
                            LinesUtil.BR(maxy + i, height),
                            LinesUtil.BC(maxx + j, width),
                        ] = 0

        if self.extend_line:
            self.extend_lines(ridge_data, label, filtered_data)

        # Adjust angles to point to the right of the line
        for contour in ridge_data.contours:
            if len(contour) > 1:
                k = (len(contour) - 1) // 2
                dy = contour.row[k + 1] - contour.row[k]
                dx = contour.col[k + 1] - contour.col[k]
                ny = np.sin(contour.angle[k])
                nx = np.cos(contour.angle[k])

                # If angles point to the left of the line, they have to be adapted
                if ny * dx - nx * dy < 0:
                    contour.angle = np.array(
                        [(ang + np.pi) % (2 * np.pi) for ang in contour.angle]
                    )

    def compute_line_width(self, ridge_data: RidgeData, filtered_data: FilteredData):
        height = ridge_data.height
        width = ridge_data.width
        length = 2.5 * filtered_data.sigma_map
        max_length = np.ceil(length * 1.2).astype(int)
        grad = np.sqrt(filtered_data.grady**2 + filtered_data.gradx**2)

        grad_dr = convolve(grad, kernel_r, mode="mirror")
        grad_dc = convolve(grad, kernel_c, mode="mirror")
        grad_dd = convolve(grad, kernel_d, mode="mirror")
        grad_drr = convolve(grad, kernel_rr, mode="mirror")
        grad_drc = convolve(grad, kernel_rc, mode="mirror")
        grad_dcc = convolve(grad, kernel_cc, mode="mirror")

        symmetric_image = np.zeros((height, width, 2, 2), dtype=float)
        symmetric_image[..., 0, 0] = 2 * grad_drr
        symmetric_image[..., 0, 1] = grad_drc
        symmetric_image[..., 1, 0] = grad_drc
        symmetric_image[..., 1, 1] = 2 * grad_dcc
        eigvals, eigvecs = np.linalg.eigh(symmetric_image)
        idx = np.absolute(eigvals).argsort()[..., ::-1]
        eigvals = np.take_along_axis(eigvals, idx, axis=-1)
        eigvecs = np.take_along_axis(eigvecs, idx[:, :, None, :], axis=-1)

        bb = grad_dr * eigvecs[:, :, 0, 0] + grad_dc * eigvecs[:, :, 1, 0]
        aa = 2.0 * (
            grad_drr * eigvecs[:, :, 0, 0] ** 2
            + grad_drc * eigvecs[:, :, 0, 0] * eigvecs[:, :, 1, 0]
            + grad_dcc * eigvecs[:, :, 1, 0] ** 2
        )
        tt = bb / (aa + np.finfo(float).eps)
        pp1, pp2 = tt * eigvecs[:, :, 0, 0], tt * eigvecs[:, :, 1, 0]
        grad_rl = (
            grad_dd
            + pp1 * grad_dr
            + pp2 * grad_dc
            + pp1 * pp1 * grad_drr
            + pp1 * pp2 * grad_drc
            + pp2 * pp2 * grad_dcc
        )

        for contour in ridge_data.contours:
            num_points = len(contour)
            width_l = np.zeros(num_points, dtype=float)
            width_r = np.zeros(num_points, dtype=float)
            grad_l = np.zeros(num_points, dtype=float)
            grad_r = np.zeros(num_points, dtype=float)
            pos_x = np.zeros(num_points, dtype=float)
            pos_y = np.zeros(num_points, dtype=float)

            for j in range(num_points):
                py, px = contour.row[j], contour.col[j]
                pos_y[j], pos_x[j] = py, px
                r, c = LinesUtil.BR(round(py), height), LinesUtil.BC(round(px), width)
                ny, nx = np.sin(contour.angle[j]), np.cos(contour.angle[j])

                line = bresenham(ny, nx, max_length[r, c])
                num_line = line.shape[0]
                width_r[j] = width_l[j] = 0

                for direct in [-1, 1]:
                    for k in range(num_line):
                        y, x = (
                            LinesUtil.BR(r + direct * line[k, 0], height),
                            LinesUtil.BC(c + direct * line[k, 1], width),
                        )
                        val = -eigvals[y, x, 0]
                        if val > 0.0:
                            p1, p2 = pp1[y, x], pp2[y, x]

                            if abs(p1) <= 0.5 and abs(p2) <= 0.5:
                                t = ny * (py - (r + direct * line[k, 0] + p1)) + nx * (
                                    px - (c + direct * line[k, 1] + p2)
                                )
                                if direct == 1:
                                    grad_r[j] = grad_rl[y, x]
                                    width_r[j] = abs(t)
                                else:
                                    grad_l[j] = grad_rl[y, x]
                                    width_l[j] = abs(t)
                                break
            fix_locations(
                contour,
                width_l,
                width_r,
                grad_l,
                grad_r,
                pos_y,
                pos_x,
                filtered_data.sigma_map,
                self.correct_pos,
                self.mode,
            )
        return ridge_data

    def prune_contours(self, ridge_data: RidgeData) -> RidgeData:
        if self.min_len <= 0:
            return ridge_data

        id_remove = []
        pruned_contours = []
        for i in range(len(ridge_data.contours)):
            cont_len = ridge_data.contours[i].estimate_length()
            if cont_len < self.min_len or (0 < self.max_len < cont_len):
                id_remove.append(ridge_data.contours[i].id)
            else:
                pruned_contours.append(ridge_data.contours[i])
        pruned_junctions = [
            j
            for j in ridge_data.junctions
            if j.cont1 not in id_remove and j.cont2 not in id_remove
        ]
        ridge_data.contours = pruned_contours
        ridge_data.junctions = pruned_junctions
        return ridge_data

    def detect_lines(self, image):
        image = iio.imread(image) if isinstance(image, (str, Path)) else image
        data = RidgeData(image=image)

        filtered_data = self.apply_filtering(data)
        line_points = self.compute_line_points(data, filtered_data)
        self.compute_contours(data, filtered_data, line_points)
        if self.estimate_width:
            data = self.compute_line_width(data, filtered_data)
        data = self.prune_contours(data)
        self.data = data
        return self.data

    def save_results(
        self,
        save_dir=None,
        prefix="",
        make_binary=True,
        draw_junc=False,
        draw_width=True,
    ):
        if self.data is None:
            raise ValueError("Ridge data is not initialized.")
        return self.data.export_images(
            save_dir=save_dir,
            prefix=prefix,
            make_binary=make_binary,
            draw_junc=draw_junc,
            draw_width=draw_width and self.estimate_width,
        )

    def show_results(self, figsize=16, show=True):
        if self.data is None:
            raise ValueError("Ridge data is not initialized.")
        if self.estimate_width:
            fig, axes = plt.subplots(2, 2, figsize=(figsize, figsize))
            axes[0, 0].imshow(self.data.get_image_contours(show_width=False))
            axes[0, 0].set_title("contours")
            axes[0, 1].imshow(self.data.get_image_contours(show_width=True))
            axes[0, 1].set_title("contours and widths")
            axes[1, 0].imshow(self.data.get_binary_contours())
            axes[1, 0].set_title("binary contours")
            axes[1, 1].imshow(self.data.get_binary_widths())
            axes[1, 1].set_title("binary widths")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(figsize, figsize / 2))
            axes[0].imshow(self.data.get_image_contours(show_width=False))
            axes[0].set_title("contours")
            axes[1].imshow(self.data.get_binary_contours())
            axes[1].set_title("binary contours")
        if show:
            plt.show()


if __name__ == "__main__":
    detector = RidgeDetector(
        line_widths=np.arange(7, 11),
        low_contrast=50,
        high_contrast=100,
        min_len=15,
        max_len=0,
        dark_line=False,
        estimate_width=True,
        extend_line=True,
        correct_pos=False,
    )

    detector.detect_lines(Path(__file__).parent.parent / "data/images/img2.jpg")
    detector.show_results()
    # plt.imshow(detector.data.eigvals)
    # plt.show()
    # detector.save_results("../data/results/", prefix="img7")
