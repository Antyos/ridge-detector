import colorsys
import math
import random
from collections.abc import Sequence
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy.typing import DTypeLike, NDArray
from scipy.ndimage import convolve, gaussian_filter1d

from .constants import LINE_WIDTH_COMPENSATION
from .correct import Correct

ArrayLikeInt = int | np.integer | NDArray[np.integer] | Sequence[int]
ArrayLikeFloat = float | np.floating | NDArray[np.floating] | Sequence[float]


class LineData:
    row: list[int]
    col: list[int]
    angle: list[float]
    response: list[float]

    def __init__(self, row=None, col=None, angle=None, resp=None):
        self.row = row if row is not None else []
        self.col = col if col is not None else []
        self.angle = angle if angle is not None else []
        self.response = resp if resp is not None else []

    def append(self, row, col, angle, response):
        self.row.append(row)
        self.col.append(col)
        self.angle.append(angle)
        self.response.append(response)

    def reverse(self):
        """Reverse the order of points in the line data."""
        self.row.reverse()
        self.col.reverse()
        self.angle.reverse()
        self.response.reverse()

    def __len__(self):
        return len(self.row)

    def to_line(self, contour_class: Optional["LinesUtil.ContourClass"] = None):
        """Convert LineData to Line object."""
        return Line(
            row=self.row,
            col=self.col,
            angle=self.angle,
            response=self.response,
            contour_class=contour_class,
        )


class LineView:
    """Data structure to hold point information."""

    def __init__(self, line: "Line", index: int | slice):
        self.line = line
        self.index = index

    @property
    def x(self):
        return self.line.col[self.index]

    @property
    def y(self):
        return self.line.row[self.index]

    @property
    def angle(self):
        return self.line.angle[self.index]

    @property
    def response(self):
        return self.line.response[self.index]

    @property
    def width_l(self):
        return self.line.width_l[self.index]

    @property
    def width_r(self):
        return self.line.width_r[self.index]

    @property
    def asymmetry(self):
        return self.line.asymmetry[self.index]

    @property
    def intensity(self):
        return self.line.intensity[self.index]

    @property
    def contour_class(self):
        return self.line.contour_class


class Line:
    id_counter = 0  # Class variable for tracking ID

    row: NDArray[np.integer]
    col: NDArray[np.integer]
    angle: NDArray[np.floating]
    response: NDArray[np.floating]
    width_l: NDArray[np.floating]
    width_r: NDArray[np.floating]
    asymmetry: NDArray[np.floating]
    intensity: NDArray[np.floating]

    def __init__(
        self,
        row: ArrayLikeFloat | ArrayLikeInt,
        col: ArrayLikeFloat | ArrayLikeInt,
        angle: ArrayLikeFloat,
        response: ArrayLikeFloat,
        width_l: Optional[ArrayLikeFloat] = None,
        width_r: Optional[ArrayLikeFloat] = None,
        asymmetry: Optional[ArrayLikeFloat] = None,
        intensity: Optional[ArrayLikeFloat] = None,
        contour_class: Optional["LinesUtil.ContourClass"] = None,
    ):
        self.id = Line.id_counter
        Line.id_counter += 1

        self.row = np.array(row)
        self.col = np.array(col)
        self.angle = np.array(angle)
        self.response = np.array(response)
        length = len(self.row)
        self.width_l = np.array(width_l) or np.zeros(length, dtype=np.float32)
        self.width_r = np.array(width_r) or np.zeros(length, dtype=np.float32)
        self.asymmetry = np.array(asymmetry) or np.zeros(length, dtype=np.float32)
        self.intensity = np.array(intensity) or np.zeros(length, dtype=np.float32)
        self.contour_class = contour_class

    def __len__(self):
        return len(self.row)

    def __iter__(self):
        return (LineView(self, i) for i in range(len(self)))

    def __getitem__(self, index: int | slice):
        """Get a point by index."""
        return LineView(self, index)

    @property
    def num(self):
        return len(self)

    def get_contour_class(self):
        return self.contour_class

    def get_contour_class_str(self):
        if self.contour_class:
            # Return the name of the contour class without the prefix "cont_"
            return self.contour_class.name[self.contour_class.name.find("_") + 1 :]
        return "none"

    def set_contour_class(self, contour_class):
        self.contour_class = contour_class

    def get_x_coordinates(self):
        return self.col

    def get_y_coordinates(self):
        return self.row

    def get_response(self):
        return self.response

    def get_intensity(self):
        return self.intensity

    def get_angle(self):
        return self.angle

    def get_asymmetry(self):
        return self.asymmetry

    def get_line_width_l(self):
        return self.width_l

    def get_line_width_r(self):
        return self.width_r

    def get_number(self):
        return self.num

    def get_id(self):
        return self.id

    def get_line_class(self):
        return self.get_contour_class()

    def get_start_or_end_position(self, x, y):
        dist_start = ((self.col[0] - x) ** 2 + (self.row[0] - y) ** 2) ** 0.5
        dist_end = ((self.col[-1] - x) ** 2 + (self.row[-1] - y) ** 2) ** 0.5
        return 0 if dist_start < dist_end else self.num - 1

    def estimate_length(self) -> float:
        """Get the total length of the contour."""
        col_diff = np.diff(self.col)
        row_diff = np.diff(self.row)
        length = np.sum(np.sqrt(col_diff**2 + row_diff**2))
        return length

    def estimate_width(self) -> NDArray[np.floating]:
        """Estimate the width of the line at each segment."""
        if self.width_l is not None and self.width_r is not None:
            return self.width_l + self.width_r
        return np.zeros(self.num, dtype=np.float32)

    @classmethod
    def reset_counter(cls):
        cls.id_counter = 0


class Crossref:
    def __init__(self, y=0, x=0, value=0.0, done=False):
        self.y = y
        self.x = x
        self.value = value
        self.done = done

    def __lt__(self, other):
        # Reverse the logic for descending order
        return self.value > other.value


class LinesUtil:
    class DERIV(Enum):
        """Enum for derivative types."""

        R = 1  # Derivative in row direction
        C = 2  # Derivative in column direction
        RR = 3  # Second derivative in row direction
        RC = 4  # Second derivative in row and column direction
        CC = 5  # Second derivative in column direction

    class MODE(Enum):
        """Enum for line extraction modes."""

        light = -1  # Extract bright lines
        dark = 1  # Extract dark lines

    MAX_SIZE_MASK_0 = 3.09023230616781  # Size for Gaussian mask
    MAX_SIZE_MASK_1 = 3.46087178201605  # Size for 1st derivative mask
    MAX_SIZE_MASK_2 = 3.82922419517181  # Size for 2nd derivative mask

    @staticmethod
    def MASK_SIZE(MAX, sigma):
        return int(MAX * sigma + 0.5)  # Maximum mask index

    @staticmethod
    def LINCOOR(row, col, width):
        return row * width + col

    @staticmethod
    def BR(row, height):
        return (
            np.abs(row)
            if row < 0
            else (height - row + height - 2)
            if row >= height
            else row
        )

    @staticmethod
    def BC(col, width):
        return (
            np.abs(col)
            if col < 0
            else (width - col + width - 2)
            if col >= width
            else col
        )

    class ContourClass(Enum):
        # The cont no junc
        cont_no_junc = (1,)
        # The cont start junc
        # no end point is a junction
        cont_start_junc = (2,)
        # The cont end junc.
        # only the start point of the line is a junction
        cont_end_junc = (3,)
        # The cont both junc.
        # only the end point of the line is a junction
        cont_both_junc = (4,)
        # The cont closed.
        # both end points of the line are junctions
        cont_closed = 5  # the contour is closed


class Junction:
    cont1: int
    cont2: int
    pos: int
    y: float
    x: float

    def __init__(
        self,
        cont1: int = -1,
        cont2: int = -1,
        pos: int = 0,
        y: float = 0.0,
        x: float = 0.0,
        line_cont1=None,
        line_cont2=None,
        is_non_terminal=False,
    ):
        self.cont1 = cont1
        self.cont2 = cont2
        self.pos = pos
        self.y = y
        self.x = x
        self.line_cont1 = line_cont1
        self.line_cont2 = line_cont2
        self.is_non_terminal = is_non_terminal

    def __lt__(self, other):
        """Implements less than for sorting. Compares first by cont1, then by pos."""
        if self.cont1 != other.cont1:
            return self.cont1 < other.cont1
        else:
            return self.pos < other.pos


class Normal:
    SQRT_2_PI_INV = 0.39894228040143267793994605993
    SQRTPI = 1.772453850905516027
    UPPERLIMIT = 20.0

    P10 = 242.66795523053175
    P11 = 21.979261618294152
    P12 = 6.9963834886191355
    P13 = -0.035609843701815385

    Q10 = 215.05887586986120
    Q11 = 91.164905404514901
    Q12 = 15.082797630407787
    Q13 = 1.0

    P20 = 300.4592610201616005
    P21 = 451.9189537118729422
    P22 = 339.3208167343436870
    P23 = 152.9892850469404039
    P24 = 43.16222722205673530
    P25 = 7.211758250883093659
    P26 = 0.5641955174789739711
    P27 = -0.0000001368648573827167067

    Q20 = 300.4592609569832933
    Q21 = 790.9509253278980272
    Q22 = 931.3540948506096211
    Q23 = 638.9802644656311665
    Q24 = 277.5854447439876434
    Q25 = 77.00015293522947295
    Q26 = 12.78272731962942351
    Q27 = 1.0

    P30 = -0.00299610707703542174
    P31 = -0.0494730910623250734
    P32 = -0.226956593539686930
    P33 = -0.278661308609647788
    P34 = -0.0223192459734184686

    Q30 = 0.0106209230528467918
    Q31 = 0.191308926107829841
    Q32 = 1.05167510706793207
    Q33 = 1.98733201817135256
    Q34 = 1.0

    SQRT2 = 1.41421356237309504880

    @staticmethod
    def getNormal(x: NDArray):
        if x < -Normal.UPPERLIMIT:
            return 0.0
        if x > Normal.UPPERLIMIT:
            return 1.0

        y = x / Normal.SQRT2
        sn = 1
        if y < 0:
            y = -y
            sn = -1

        y2 = y * y
        y4 = y2 * y2
        y6 = y4 * y2

        if y < 0.46875:
            R1 = Normal.P10 + Normal.P11 * y2 + Normal.P12 * y4 + Normal.P13 * y6
            R2 = Normal.Q10 + Normal.Q11 * y2 + Normal.Q12 * y4 + Normal.Q13 * y6
            erf = y * R1 / R2
            if sn == 1:
                phi = 0.5 + 0.5 * erf
            else:
                phi = 0.5 - 0.5 * erf
        elif y < 4.0:
            y3 = y2 * y
            y5 = y4 * y
            y7 = y6 * y
            R1 = (
                Normal.P20
                + Normal.P21 * y
                + Normal.P22 * y2
                + Normal.P23 * y3
                + Normal.P24 * y4
                + Normal.P25 * y5
                + Normal.P26 * y6
                + Normal.P27 * y7
            )
            R2 = (
                Normal.Q20
                + Normal.Q21 * y
                + Normal.Q22 * y2
                + Normal.Q23 * y3
                + Normal.Q24 * y4
                + Normal.Q25 * y5
                + Normal.Q26 * y6
                + Normal.Q27 * y7
            )
            erfc = np.exp(-y2) * R1 / R2
            if sn == 1:
                phi = 1.0 - 0.5 * erfc
            else:
                phi = 0.5 * erfc
        else:
            z = y4
            z2 = z * z
            z3 = z2 * z
            z4 = z2 * z2
            R1 = (
                Normal.P30
                + Normal.P31 * z
                + Normal.P32 * z2
                + Normal.P33 * z3
                + Normal.P34 * z4
            )
            R2 = (
                Normal.Q30
                + Normal.Q31 * z
                + Normal.Q32 * z2
                + Normal.Q33 * z3
                + Normal.Q34 * z4
            )
            erfc = (np.exp(-y2) / y) * (1.0 / Normal.SQRTPI + R1 / (R2 * y2))
            if sn == 1:
                phi = 1.0 - 0.5 * erfc
            else:
                phi = 0.5 * erfc

        return phi


def phi0(x, sigma):
    return Normal.getNormal(x / sigma)


def phi1(x, sigma):
    t = x / sigma
    return Normal.SQRT_2_PI_INV / sigma * np.exp(-0.5 * t * t)


def phi2(x, sigma):
    t = x / sigma
    return -x * Normal.SQRT_2_PI_INV / (sigma**3.0) * np.exp(-0.5 * t * t)


def compute_gauss_mask_0(sigma):
    limit = LinesUtil.MASK_SIZE(LinesUtil.MAX_SIZE_MASK_0, sigma)
    n = int(limit)
    h = np.zeros(2 * n + 1, dtype=float)
    for i in range(-n + 1, n):
        h[n + i] = phi0(-i + 0.5, sigma) - phi0(-i - 0.5, sigma)
    h[0] = 1.0 - phi0(n - 0.5, sigma)
    h[2 * n] = phi0(-n + 0.5, sigma)
    return h, n


def compute_gauss_mask_1(sigma):
    limit = LinesUtil.MASK_SIZE(LinesUtil.MAX_SIZE_MASK_1, sigma)
    n = int(limit)
    h = np.zeros(2 * n + 1, dtype=float)
    for i in range(-n + 1, n):
        h[n + i] = phi1(-i + 0.5, sigma) - phi1(-i - 0.5, sigma)
    h[0] = -phi1(n - 0.5, sigma)
    h[2 * n] = phi1(-n + 0.5, sigma)
    return h, n


def compute_gauss_mask_2(sigma):
    limit = LinesUtil.MASK_SIZE(LinesUtil.MAX_SIZE_MASK_2, sigma)
    n = int(limit)
    h = np.zeros(2 * n + 1, dtype=float)
    for i in range(-n + 1, n):
        h[n + i] = phi2(-i + 0.5, sigma) - phi2(-i - 0.5, sigma)
    h[0] = -phi2(n - 0.5, sigma)
    h[2 * n] = phi2(-n + 0.5, sigma)
    return h, n


def convolve_gauss(image: NDArray, sigma: float, deriv_type: LinesUtil.DERIV):
    if deriv_type == LinesUtil.DERIV.R:
        hr, nr = compute_gauss_mask_1(sigma)
        hc, nc = compute_gauss_mask_0(sigma)
    elif deriv_type == LinesUtil.DERIV.C:
        hr, nr = compute_gauss_mask_0(sigma)
        hc, nc = compute_gauss_mask_1(sigma)
    elif deriv_type == LinesUtil.DERIV.RR:
        hr, nr = compute_gauss_mask_2(sigma)
        hc, nc = compute_gauss_mask_0(sigma)
    elif deriv_type == LinesUtil.DERIV.RC:
        hr, nr = compute_gauss_mask_1(sigma)
        hc, nc = compute_gauss_mask_1(sigma)
    elif deriv_type == LinesUtil.DERIV.CC:
        hr, nr = compute_gauss_mask_0(sigma)
        hc, nc = compute_gauss_mask_2(sigma)
    else:
        raise ValueError("Invalid derivative type")

    return convolve(
        convolve(image, hr.reshape(-1, 1), mode="nearest"),
        hc.reshape(1, -1),
        mode="nearest",
    )


def normalize(
    x, pmin=2, pmax=98, axis=None, eps=1e-20, dtype: Optional[DTypeLike] = np.float32
):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = mi.astype(dtype, copy=False)
        ma = ma.astype(dtype, copy=False)
        eps = np.array(eps, dtype=dtype).item()
    x = (x - mi) / (ma - mi + eps)

    return np.clip(x, 0, 1)


@jit(nopython=True)
def closest_point(ly, lx, dy, dx, py, px):
    my = py - ly
    mx = px - lx
    den = dy * dy + dx * dx
    nom = my * dy + mx * dx
    tt = nom / den if den != 0 else 0
    return ly + tt * dy, lx + tt * dx, tt


@jit(nopython=True)
def bresenham(ny, nx, length, py=0.0, px=0.0):
    points = []
    x, y = 0, 0
    dx, dy = abs(nx), abs(ny)
    s1 = 1 if nx > 0 else -1
    s2 = 1 if ny > 0 else -1
    px *= s1
    py *= s2
    xchg = False
    if dy > dx:
        dx, dy = dy, dx
        px, py = py, px
        xchg = True

    maxit = int(np.ceil(length * dx))
    d_err = dy / dx
    e = (0.5 - px) * dy / dx - (0.5 - py)
    for i in range(maxit + 1):
        points.append([y, x])
        while e >= -1e-8:
            if xchg:
                x += s1
            else:
                y += s2
            e -= 1
            if e > -1:
                points.append([y, x])
        if xchg:
            y += s2
        else:
            x += s1
        e += d_err
    return np.array(points)


@jit(nopython=True)
def interpolate_gradient_test(grady, gradx, py, px):
    giy, gix = math.floor(py), math.floor(px)
    gfy, gfx = py % 1.0, px % 1.0

    gy1, gx1 = grady[giy, gix], gradx[giy, gix]
    gy2, gx2 = grady[giy + 1, gix], gradx[giy + 1, gix]
    gy3, gx3 = grady[giy, gix + 1], grady[giy, gix + 1]
    gy4, gx4 = grady[giy + 1, gix + 1], gradx[giy + 1, gix + 1]

    gy = (1 - gfy) * ((1 - gfx) * gy1 + gfx * gy2) + gfy * ((1 - gfx) * gy3 + gfx * gy4)
    gx = (1 - gfy) * ((1 - gfx) * gx1 + gfx * gx2) + gfy * ((1 - gfx) * gx3 + gfx * gx4)

    return gy, gx


def fill_gaps(master, slave1, slave2, cont):
    num_points = cont.num
    i = 0
    while i < num_points:
        if master[i] == 0:
            j = i + 1
            while j < num_points and master[j] == 0:
                j += 1

            m_s, m_e, s1_s, s1_e, s2_s, s2_e = 0, 0, 0, 0, 0, 0
            if i > 0 and j < num_points - 1:
                s, e = i, j - 1
                m_s, m_e = master[s - 1], master[e + 1]
                if slave1 is not None:
                    s1_s, s1_e = slave1[s - 1], slave1[e + 1]
                if slave2 is not None:
                    s2_s, s2_e = slave2[s - 1], slave2[e + 1]
            elif i > 0:
                s, e = i, num_points - 2
                m_s, m_e = master[s - 1], master[s - 1]
                master[e + 1] = m_e
                if slave1 is not None:
                    s1_s, s1_e = slave1[s - 1], slave1[s - 1]
                    slave1[e + 1] = s1_e
                if slave2 is not None:
                    s2_s, s2_e = slave2[s - 1], slave2[s - 1]
                    slave2[e + 1] = s2_e
            elif j < num_points - 1:
                s, e = 1, j - 1
                m_s, m_e = master[e + 1], master[e + 1]
                master[s - 1] = m_s
                if slave1 is not None:
                    s1_s, s1_e = slave1[e + 1], slave1[e + 1]
                    slave1[s - 1] = s1_s
                if slave2 is not None:
                    s2_s, s2_e = slave2[e + 1], slave2[e + 1]
                    slave2[s - 1] = s2_s
            else:
                s, e = 1, num_points - 2
                m_s, m_e = master[s - 1], master[e + 1]
                if slave1 is not None:
                    s1_s, s1_e = slave1[s - 1], slave1[e + 1]
                if slave2 is not None:
                    s2_s, s2_e = slave2[s - 1], slave2[e + 1]

            arc_len = np.sum(
                np.sqrt(
                    np.diff(cont.row[s : e + 2]) ** 2
                    + np.diff(cont.col[s : e + 2]) ** 2
                )
            )
            if arc_len != 0.0:
                len_ = 0
                for k in range(s, e + 1):
                    d_r = cont.row[k] - cont.row[k - 1]
                    d_c = cont.col[k] - cont.col[k - 1]
                    len_ += np.sqrt(d_r * d_r + d_c * d_c)
                    master[k] = (arc_len - len_) / arc_len * m_s + len_ / arc_len * m_e
                    if slave1 is not None:
                        slave1[k] = (
                            arc_len - len_
                        ) / arc_len * s1_s + len_ / arc_len * s1_e
                    if slave2 is not None:
                        slave2[k] = (
                            arc_len - len_
                        ) / arc_len * s2_s + len_ / arc_len * s2_e
            i = j
        else:
            i += 1
    return master


def normalize_to_half_circle(angle):
    if angle < 0.0:
        angle += 2.0 * np.pi
    if angle >= np.pi:
        angle -= np.pi
    return angle


def interpolate_response(resp, x, y, px, py, width, height):
    i1 = resp[
        LinesUtil.LINCOOR(
            LinesUtil.BR(x - 1, height), LinesUtil.BC(y - 1, width), width
        )
    ]
    i2 = resp[LinesUtil.LINCOOR(LinesUtil.BR(x - 1, height), y, width)]
    i3 = resp[
        LinesUtil.LINCOOR(
            LinesUtil.BR(x - 1, height), LinesUtil.BC(y + 1, width), width
        )
    ]
    i4 = resp[LinesUtil.LINCOOR(x, LinesUtil.BC(y - 1, width), width)]
    i5 = resp[LinesUtil.LINCOOR(x, y, width)]
    i6 = resp[LinesUtil.LINCOOR(x, LinesUtil.BC(y + 1, width), width)]
    i7 = resp[
        LinesUtil.LINCOOR(
            LinesUtil.BR(x + 1, height), LinesUtil.BC(y - 1, width), width
        )
    ]
    i8 = resp[LinesUtil.LINCOOR(LinesUtil.BR(x + 1, height), y, width)]
    i9 = resp[
        LinesUtil.LINCOOR(
            LinesUtil.BR(x + 1, height), LinesUtil.BC(y + 1, width), width
        )
    ]
    t1 = i1 + i2 + i3
    t2 = i4 + i5 + i6
    t3 = i7 + i8 + i9
    t4 = i1 + i4 + i7
    t5 = i2 + i5 + i8
    t6 = i3 + i6 + i9
    d = (-i1 + 2 * i2 - i3 + 2 * i4 + 5 * i5 + 2 * i6 - i7 + 2 * i8 - i9) / 9
    dr = (t3 - t1) / 6
    dc = (t6 - t4) / 6
    drr = (t1 - 2 * t2 + t3) / 6
    dcc = (t4 - 2 * t5 + t6) / 6
    drc = (i1 - i3 - i7 + i9) / 4
    xx = px - x
    yy = py - y
    return d + xx * dr + yy * dc + xx**2 * drr + xx * yy * drc + yy**2 * dcc


def interpolate_gradient(
    gradx: NDArray, grady: NDArray, px: float, py: float, width: int
):
    gix = int(px // 1)
    giy = int(py // 1)
    gfx = px % 1.0
    gfy = py % 1.0

    gpos = LinesUtil.LINCOOR(gix, giy, width)
    gx1, gy1 = gradx[gpos], grady[gpos]
    gpos = LinesUtil.LINCOOR(gix + 1, giy, width)
    gx2, gy2 = gradx[gpos], grady[gpos]
    gpos = LinesUtil.LINCOOR(gix, giy + 1, width)
    gx3, gy3 = gradx[gpos], grady[gpos]
    gpos = LinesUtil.LINCOOR(gix + 1, giy + 1, width)
    gx4, gy4 = gradx[gpos], grady[gpos]

    gx = (1 - gfy) * ((1 - gfx) * gx1 + gfx * gx2) + gfy * ((1 - gfx) * gx3 + gfx * gx4)
    gy = (1 - gfy) * ((1 - gfx) * gy1 + gfx * gy2) + gfy * ((1 - gfx) * gy3 + gfx * gy4)
    return gx, gy


def fix_locations(
    cont: Line,
    width_l,
    width_r,
    grad_l,
    grad_r,
    pos_y,
    pos_x,
    sigma_map,
    correct_pos=True,
    mode=LinesUtil.MODE.dark,
):
    num_points = cont.num
    correction = np.zeros(num_points, dtype=float)
    asymm = np.zeros(num_points, dtype=float)

    # Fill gaps in width_l and width_r
    fill_gaps(width_l, grad_l, None, cont)
    fill_gaps(width_r, grad_r, None, cont)

    # Correct positions if required
    if correct_pos:
        correct_start = (
            cont.contour_class
            in [
                LinesUtil.ContourClass.cont_no_junc,
                LinesUtil.ContourClass.cont_end_junc,
                LinesUtil.ContourClass.cont_closed,
            ]
        ) and (width_r[0] > 0 and width_l[0] > 0)
        correct_end = (
            cont.contour_class
            in [
                LinesUtil.ContourClass.cont_no_junc,
                LinesUtil.ContourClass.cont_start_junc,
                LinesUtil.ContourClass.cont_closed,
            ]
        ) and (width_r[-1] > 0 and width_l[-1] > 0)

        for i in range(num_points):
            if width_r[i] > 0 and width_l[i] > 0:
                w_est = (width_r[i] + width_l[i]) * LINE_WIDTH_COMPENSATION
                if grad_r[i] <= grad_l[i]:
                    r_est = grad_r[i] / grad_l[i]
                    weak_is_r = True
                else:
                    r_est = grad_l[i] / grad_r[i]
                    weak_is_r = False
                sigma = sigma_map[int(cont.row[i]), int(cont.col[i])]
                w_real, h_real, corr, w_strong, w_weak = Correct.line_corrections(
                    sigma, w_est, r_est
                )
                w_real /= LINE_WIDTH_COMPENSATION
                corr /= LINE_WIDTH_COMPENSATION
                width_r[i], width_l[i] = w_real, w_real
                if weak_is_r:
                    asymm[i] = h_real
                    correction[i] = -corr
                else:
                    asymm[i] = -h_real
                    correction[i] = corr

        fill_gaps(width_l, correction, asymm, cont)
        width_r = width_l[:]

        if not correct_start:
            correction[0] = 0
        if not correct_end:
            correction[-1] = 0

        for i in range(num_points):
            py, px = pos_y[i], pos_x[i]
            ny, nx = np.sin(cont.angle[i]), np.cos(cont.angle[i])
            px += correction[i] * nx
            py += correction[i] * ny
            pos_y[i], pos_x[i] = py, px

    # Update position of line and add extracted width
    width_l = gaussian_filter1d(width_l, 3.0, mode="mirror")
    width_r = gaussian_filter1d(width_r, 3.0, mode="mirror")
    cont.width_l = np.array([float(w) for w in width_l])
    cont.width_r = np.array([float(w) for w in width_r])
    cont.row = np.array([float(y) for y in pos_y])
    cont.col = np.array([float(x) for x in pos_x])

    return cont


def color_line_segments(image: NDArray, conts: Sequence[Line]):
    for cont in conts:
        hue = random.random()
        saturation = 1
        brightness = 1
        # Convert HSV color to RGB color
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
        # Scale RGB values to the range [0, 255]
        random_color = [int(r * 255), int(g * 255), int(b * 255)]
        for j in range(cont.num):
            image[round(cont.row[j]), round(cont.col[j]), :] = random_color

    return image


def visualize(gray, mag, ny, nx, saliency, gd=5):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    h00 = axes[0, 0].imshow(gray)
    h01 = axes[0, 1].imshow(mag)
    h10 = axes[1, 0].imshow(saliency * 127)
    axes[1, 1].imshow(gray)
    rows, cols = saliency.shape
    indices = np.argwhere(saliency)
    row_idx, col_idx = indices[:, 0], indices[:, 1]
    Y, X = np.mgrid[0:rows, 0:cols]
    S = np.zeros_like(mag)
    S[row_idx, col_idx] = mag[row_idx, col_idx]

    axes[1, 1].quiver(
        X[::gd, ::gd],
        Y[::gd, ::gd],
        (S * nx)[::gd, ::gd],
        (S * ny)[::gd, ::gd],
        angles="xy",
        scale=100,
    )

    fig.colorbar(h00, ax=axes[0, 0])
    fig.colorbar(h01, ax=axes[0, 1])
    fig.colorbar(h10, ax=axes[1, 0])
    axes[0, 0].set_title("Gray image")
    axes[0, 1].set_title("Eigen magnitude")
    axes[1, 0].set_title("Saliency map")
    axes[1, 1].set_title("Eigen vectors")
    plt.show()
