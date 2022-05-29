import abc
import functools
import gc
import json
import math

import numpy as np


def scale_img(img, h, w):
    shape = img.shape
    scale = h / shape[0]

    coords = [[{} for _ in range(w)] for _ in range(h)]

    for i in range(shape[0]):
        for j in range(shape[1]):
            x = math.floor(i * scale)
            y = math.floor(j * scale)
            elem = img[i][j]
            if len(img.shape) <= 2:
                key = elem
            else:
                key = tuple(elem)
            dicti = coords[x][y]
            if key not in dicti:
                dicti[key] = 1
            else:
                dicti[key] += 1

    if len(shape) > 2:
        new_shape = (h, w, shape[2])
    else:
        new_shape = (h, w)
    out = np.empty(new_shape)
    for i in range(h):
        for j in range(w):
            out[i][j] = functools.reduce(lambda a, b: a if a[1] > b[1] else b, coords[i][j].items())[0]

    return out


def bound_number(x, a, b):
    if b < a:
        a, b = b, a
    if x < a:
        x = a
    elif x >= b:
        x = b - 1
    return x


class Debugger:
    @abc.abstractmethod
    def resolve_print(self, key, string):
        pass

    @abc.abstractmethod
    def resolve_func(self, key, func):
        pass

    @abc.abstractmethod
    def resolve_value(self, key, default):
        pass


class DummyDebugger(Debugger):
    def __init__(self, *args, **kwargs):
        pass

    def resolve_print(self, key, string):
        pass

    def resolve_func(self, key, func):
        pass

    def resolve_value(self, key, default):
        pass


class SegmentsDebugger(Debugger):
    def __init__(self, debug_map=None):
        if debug_map is None:
            self.__debug = {
                "PRINT_SCALING": False,
                "PRINT_CREATED_LINES": False,
                "PRINT_GEN_LINES": False,
                "PRINT_GEN_MID_POINT": False,
                "PRINT_CONDENSED_LINES": False,
                "PRINT_ALL_CONDENSED_LINES": False,
                "PRINT_ATTACHED_LINES": False,
                "SHOW_MID_POINT_STEP": False,
                "SHOW_POLYGON": False,
                "SHOW_SEGMENT": False,
                "SHOW_PAINT_STEP": False,
                "SHOW_PAINT_FINAL": False,

                "FIRST_SEGMENT": 0,
                "START_IMG": 0
            }

    def resolve_print(self, key, string):
        if self.__debug[key]:
            print(string)

    def resolve_func(self, key, func):
        if self.__debug[key]:
            func()

    def resolve_value(self, key, default):
        if key in self.__debug.keys():
            return self.__debug[key]
        return default


class Debuggable:
    def __init__(self):
        super(Debuggable, self).__init__()
        self._debugger = DummyDebugger()

    def apply_debugger(self, debugger: Debugger):
        self._debugger = debugger


class Shape:
    def __init__(self, w, h):
        self.W = w
        self.H = h

    def scale(self, scale):
        self.W = math.ceil(self.W * scale)
        self.H = math.ceil(self.H * scale)


class ShapeDependable:
    def __init__(self):
        super(ShapeDependable, self).__init__()
        self._shape = None

    def apply_shape(self, shape: Shape):
        self._shape = shape


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def coords(self):
        return self.x, self.y

    def dist(self, point):
        x1, y1 = point.coords()
        dist = math.sqrt((y1 - self.y) ** 2 + (x1 - self.x) ** 2)
        return dist

    def __str__(self):
        return f"({self.x}, {self.y})"


class PointLine(Debuggable):
    def __init__(self, p1: Point, p2: Point):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.__len = -1
        self.__x1, self.__y1 = p1.coords()
        self.__x2, self.__y2 = p2.coords()
        self.is_from_left = False
        self.__min_x, self.__max_x = None, None
        self.__min_y, self.__max_y = None, None
        self.__calc_params()

    @property
    def x1(self):
        return self.__x1

    @property
    def x2(self):
        return self.__x2

    @property
    def y1(self):
        return self.__y1

    @property
    def y2(self):
        return self.__y2

    @property
    def min_x(self):
        return self.__min_x

    @property
    def min_y(self):
        return self.__min_y

    @property
    def max_x(self):
        return self.__max_x

    @property
    def max_y(self):
        return self.__max_y

    def set_last_point(self, x, y):
        self.__set_second_point(x, y)
        self.__calc_params()

    def is_point_in_bounds(self, point: Point):
        return self.__min_x <= point.x <= self.__max_x and self.__max_y >= point.y >= self.__min_y

    def length(self):
        return self.__len

    def __str__(self):
        return f"from_left:{self.is_from_left}, length:{self.length()}, ({self.x1:.2f}, {self.y1:.2f}) -> ({self.x2:.2f}, {self.y2:.2f})"

    def __calc_params(self):
        if self.__x2 > self.__x1:
            self.is_from_left = True
        elif self.__x1 > self.__x2:
            self.is_from_left = False
        else:
            if self.__y2 > self.__y1:
                self.is_from_left = True
            else:
                self.is_from_left = False
        self.__len = self.p1.dist(self.p2)
        self.__min_x, self.__max_x = (self.__x1, self.__x2) if self.__x1 < self.__x2 else (self.__x2, self.__x1)
        self.__min_y, self.__max_y = (self.__y1, self.__y2) if self.__y1 < self.__y2 else (self.__y2, self.__y1)

    def __set_first_point(self, x, y):
        self.p1.x = x
        self.p1.y = y
        self.__x1 = x
        self.__y1 = y

    def __set_second_point(self, x, y):
        self.p2.x = x
        self.p2.y = y
        self.__x2 = x
        self.__y2 = y

    def scale(self, scale):
        x1_s = self.x1 * scale
        y1_s = self.y1 * scale
        x2_s = self.x2 * scale
        y2_s = self.y2 * scale
        print(self)
        self._debugger.resolve_print(
            "PRINT_SCALING",
            f"[({self.x1:.2f}, {self.y1:.2f}) -> "
            f"({self.x2:.2f}, {self.y2:.2f}) ] * {scale:.2f} = [({x1_s:.2f}, {y1_s:.2f}) -> ({x2_s:.2f}, {y2_s:.2f}) ]")

        self.__set_first_point(x1_s, y1_s)
        self.__set_second_point(x2_s, y2_s)
        self.__calc_params()


class Line(ShapeDependable, Debuggable):
    def __init__(self, pline, upper=False, index=None, direction=None):
        super().__init__()
        self.pline: PointLine = pline
        self.upper = upper

        self.index = index
        self.direction = direction
        self.__next_index = self.index + self.direction
        self.m = None
        self.b = None
        self.__calc_params()

    def __str__(self):
        return f"---- m: {self.m:.2f}, b: {self.b:.2f}, upper: {self.upper}, from_left:{self.pline.is_from_left}," \
               f" length:{self.length()}, ({self.pline.x1:.2f}, {self.pline.y1:.2f}) -> ({self.pline.x2:.2f}," \
               f" {self.pline.y2:.2f}) ----"

    def __call__(self, *args, **kwargs):
        x = args[0]
        if abs(self.m) == np.inf:
            return self.pline.x1
        return self.m * x + self.b

    def apply_debugger(self, debugger: SegmentsDebugger):
        super(Line, self).apply_debugger(debugger)
        self.pline.apply_debugger(debugger)

    def scale(self, scale):
        self.pline.scale(scale)
        self.__calc_params()

    def inv_call(self, y):
        if abs(self.m) == np.inf:
            return self.pline.x1
        if self.m == 0:
            raise Exception(f"Can not call inverse function of line with: {self.m} slope value.")
        x = (y - self.b) / self.m
        return x

    def length(self):
        return self.pline.length()

    def is_acute_angle(self, line):
        return abs(self.angle_between(line)) < 1.57 and \
               not self.same_slope_sign(line) and \
               is_inverting_line(self.pline, line.pline)

    def angle_between(self, line):
        return self.theta - line.theta

    def same_slope_sign(self, line):
        return line.m > 0 and self.m > 0 or line.m < 0 and self.m < 0 or line.m == self.m

    def theta_str(self):
        return f"[m: {self.m} - th: {self.theta}]"

    def generate_mid_point_in(self, max_h=None, max_w=None):
        if max_h is None:
            max_h = self._shape.H
        if max_w is None:
            max_w = self._shape.W

        x1 = self.pline.x1
        y1 = self.pline.y1

        x2 = self.pline.x2
        y2 = self.pline.y2
        if x1 == x2:
            x_mid = x1
        else:
            x_mid = (x1 + x2) / 2
        if y1 == y2:
            y_mid = y1
        else:
            y_mid = (y1 + y2) / 2

        if abs(self.m) == np.inf:
            if self.upper:
                x = x_mid - 1
            else:
                x = x_mid + 1
            y = y_mid
        elif abs(self.m) == 0:
            if self.upper:
                y = y_mid + 1
            else:
                y = y_mid - 1
            x = x_mid
        else:
            m_n, b_n = self.get_normal_line_params(Point(x_mid, y_mid))

            dist = math.sqrt(2)

            a = m_n ** 2 + 1
            b = 2 * m_n * b_n - 2 * m_n * y_mid - 2 * x_mid
            c = x_mid ** 2 + y_mid ** 2 - 2 * b_n * y_mid + b_n ** 2 - dist ** 2

            a2 = a * 2
            if (self.upper or self.m > 0) and not (self.upper and self.m > 0):
                x = -b / a2 + math.sqrt((b ** 2) / (4 * a ** 2) - c / a)
            else:
                x = -b / a2 - math.sqrt((b ** 2) / (4 * a ** 2) - c / a)

            y = m_n * x + b_n

        if x >= max_h:
            x = max_h - 1
        if x < 0:
            x = 0
        if y >= max_w:
            y = max_w - 1
        if y < 0:
            y = 0

        mid_point = Point(int(x), int(y))

        self._debugger.resolve_print("PRINT_GEN_MID_POINT", f"MID POINT: {mid_point} on line: {self}")

        return mid_point

    def is_point_in(self, point: Point):
        if self.m == np.inf:
            is_in = False
            if point.x > self.pline.x1:
                is_in = True

            return is_in and self.upper

        x, y = point.coords()
        x_n = (y - self.b) / self.m

        is_in = False
        if (self.m > 0 and x < x_n) or (self.m < 0 and x > x_n):
            is_in = True
        if not self.upper:
            return not is_in
        return is_in

    @staticmethod
    def __same_as_prev_pt(x, y, points):
        if len(points) == 0:
            return False
        prev = points[-1]
        return prev.x == x and prev.y == y

    def generate_points_on_line(self, max_h=None, max_w=None):
        if max_h is None:
            max_h = self._shape.H
        if max_w is None:
            max_w = self._shape.W

        self._debugger.resolve_print("PRINT_GEN_LINES", f"GEN: {self}")

        x1 = self.pline.x1
        y1 = self.pline.y1

        x2 = self.pline.x2
        y2 = self.pline.y2

        x_min, x_max = (x1, x2) if x1 < x2 else (x2, x1)
        y_min, y_max = (y1, y2) if y1 < y2 else (y2, y1)

        x_min = bound_number(x_min, 0, max_h)
        x_max = bound_number(x_max, 0, max_h)
        y_min = bound_number(y_min, 0, max_w)
        y_max = bound_number(y_max, 0, max_w)

        rate = abs(self.m)
        points: [Point] = []

        if rate > 1:

            if rate == np.inf:
                rate = 1
            else:
                rate = 1 / rate
            y = y_min
            while True:

                py = int(y)
                px = int(self.inv_call(py))
                # print(f"*[{self.inv_call(py)}, {y}] * [{px}, {py}]")
                if px >= max_h:
                    px = max_h - 1

                if not self.__same_as_prev_pt(px, py, points):
                    p = Point(px, py)
                    points.append(p)
                    self._debugger.resolve_print("PRINT_GEN_LINES", f"[{self.inv_call(py)}, {y}] * [{px}, {py}]")
                if not (0 <= y <= y_max):
                    break
                y += rate
                if not (0 <= y < max_w):
                    break

        else:
            if rate < 1e-10:
                rate = 1
            x = x_min
            # print(f"({x1}, {y1}) -> ({x2}, {y2}), rate: {rate}, m: {self.m}")

            while True:
                px = int(x)

                # print(f"px:{px}, x:{x}, min_x:{x_min}, max_x:{x_max}, rate:{rate}")
                if not (px < max_h):
                    px = max_h - 1
                py = int(self(px))
                # print(f"*[{x}, {self(px)}] * [{px}, {py}]")
                if not self.__same_as_prev_pt(px, py, points):
                    p = Point(px, py)

                    # if self.m >= 0:
                    #     inc = 1
                    # else:
                    #     inc = -1

                    # if self.upper:
                    #     new_x = p.x + inc
                    #     if 0 <= new_x < H:
                    #         points.append(Point(new_x, p.y))
                    # else:
                    #     new_x = p.x - inc
                    #     if 0 <= new_x < H:
                    #         points.append(Point(new_x, p.y))
                    points.append(p)

                    self._debugger.resolve_print("PRINT_GEN_LINES", f"[{x}, {self(px)}] * [{px}, {py}]")
                if not (0 <= x <= x_max):
                    break
                x += rate
        return points

    def on_pline(self, point: Point, max_distance=2):
        if self.m == 0:
            x_n = point.x
            y_n = self.pline.y1
        elif self.m == np.inf:
            x_n = self.pline.x1
            y_n = point.y
        else:
            m_n, b_n = self.get_normal_line_params(point)
            x_n = (self.b - b_n) / (m_n - self.m)
            y_n = self.m * x_n + self.b

        p_n = Point(x_n, y_n)

        if not self.pline.is_point_in_bounds(p_n):
            return False

        dist = p_n.dist(point)
        return dist <= max_distance

    def next_index(self):
        return self.__next_index

    def get_normal_line_params(self, point):
        m_n = -1 / self.m
        b_n = -point.x * m_n + point.y

        return m_n, b_n

    def attach(self, line):
        pline1 = self.pline
        pline2 = line.pline

        x1, y1 = pline1.x1, pline1.y1
        x2, y2 = pline1.x2, pline1.y2
        x3, y3 = pline2.x1, pline2.y1
        x4, y4 = pline2.x2, pline2.y2
        self._debugger.resolve_print("PRINT_ATTACHED_LINES",
                                     f"({x1:3.3f}, {y1:3.3f}) -> ({x2:3.3f}, {y2:3.3f}) =c ({x3:3.3f}, {y3:3.3f}) -> ({x4:3.3f}, {y4:3.3f})")

        old_left = pline1.is_from_left

        pline1.set_last_point(pline2.x2, pline2.y2)

        if (old_left or pline1.is_from_left) and not (old_left and pline1.is_from_left):
            self.upper = not self.upper

        x1, y1 = pline1.x1, pline1.y1
        x2, y2 = pline1.x2, pline1.y2
        self._debugger.resolve_print("PRINT_ATTACHED_LINES", f"\t=> ({x1:3.3f}, {y1:3.3f}) -> ({x2:3.3f}, {y2:3.3f})")

    def similarity(self, line):

        if line.m == 0 and self.m == 0:
            return 1
        if line.m == 0 or self.m == 0:
            return 1 / abs(self.m - line.m)

        if abs(line.m) == abs(self.m) == np.inf:
            return 1
        if abs(line.m) == np.inf or abs(self.m) == np.inf:
            if line.m != self.m:
                return -1
            return 1

        ratio = self.m / line.m

        if ratio < 0:
            ratio = 1 + min(ratio, 1 / ratio)
        else:
            ratio = min(ratio, 1 / ratio)

        return ratio

    def __calc_params(self):
        h = self.pline.y2 - self.pline.y1
        w = self.pline.x2 - self.pline.x1
        if w == 0:
            if h < 0:
                self.m = -np.inf
            else:
                self.m = np.inf
            self.theta = 1.57
        else:
            self.m = h / w
            self.theta = math.atan(self.m)

        self.b = -self.pline.x1 * self.m + self.pline.y1


def is_first_line_to_the_right(line1: Line, line2: Line, region):
    left_1 = line1.pline.is_from_left
    left_2 = line2.pline.is_from_left

    if left_1 and not left_2:
        return True
    elif left_2 and not left_1:
        return False

    if line1.m * line2.m < 0:
        if line1.m > line2.m:
            return True
        else:

            return False
    if line1.m <= 0:
        positive_m = False
    else:
        positive_m = True
    if abs(line1.m) < abs(line2.m):
        return positive_m
    elif abs(line1.m) > abs(line2.m):
        return not positive_m
    else:
        help_line_1 = generate_line(line1.index, line1.next_index(), region)
        help_line_2 = generate_line(line2.index, line2.next_index(), region)
        return is_first_line_to_the_right(help_line_1, help_line_2, region)


def set_upper_for_right_line(line):
    if line.m >= 0:
        line.upper = True
    elif line.m < 0:
        line.upper = False
    return line


def generate_line(index, next_index, region):
    point2 = Point(region[index]["y"], region[index]["x"])
    point3 = Point(region[next_index]["y"], region[next_index]["x"])

    pline2 = PointLine(point2, point3)

    new_line = Line(pline2, index=next_index, direction=next_index - index)
    return new_line


def generate_first_line(base_index, index1, index2, region):
    line1 = generate_line(base_index, index1, region)
    line2 = generate_line(base_index, index2, region)

    # print("First two:")
    # print(line1)
    # print(line2)

    is_line1_right = is_first_line_to_the_right(line1, line2, region)
    if is_line1_right:
        set_upper_for_right_line(line1)
        return line1, 1
    else:
        set_upper_for_right_line(line2)
        return line2, -1


def gen_next_index(index, array, inc):
    index += inc
    if index >= len(array):
        index = 0
    elif index < 0:
        index = len(array) - 1
    return index


def is_inverting_line(pline1: PointLine, pline2: PointLine):
    if pline1.is_from_left == pline2.is_from_left:
        return False
    return True


class Polygon(Debuggable, ShapeDependable):
    def __init__(self, lines: [Line]):
        super().__init__()
        self.lines = lines
        self.__main = False
        self.bounds = None
        self.__calc_params()

    def apply_debugger(self, debugger: SegmentsDebugger):
        super(Polygon, self).apply_debugger(debugger)
        for line in self.lines:
            line.apply_debugger(debugger)

    def apply_shape(self, shape: Shape):
        super(Polygon, self).apply_shape(shape)
        for line in self.lines:
            line.apply_shape(shape)

    @property
    def main(self):
        return self.__main

    @main.setter
    def main(self, val):
        self.__main = val

    def invert(self):
        for line in self.lines:
            line.upper = not line.upper

    def scale(self, scale):
        for line in self.lines:
            line.scale(scale)
        self.__calc_params()

    def __calc_params(self):

        local_min_y = np.inf
        local_min_x = np.inf
        local_max_y = -1
        local_max_x = -1
        for i, line in enumerate(self.lines):
            x, y = line.pline.x1, line.pline.y1

            if x > local_max_x:
                local_max_x = x
            if y > local_max_y:
                local_max_y = y
            if x < local_min_x:
                local_min_x = x
            if y < local_min_y:
                local_min_y = y

        self.bounds = [local_min_x, local_min_y, local_max_x, local_max_y]

    def generate_center_point(self, max_h=None, max_w=None):
        if max_h is None:
            max_h = self._shape.H
        if max_w is None:
            max_w = self._shape.W

        max_line: Line = None
        max_length = -1
        for line in self.lines:
            length = line.length()

            if length > max_length:
                max_length = length
                max_line = line

        center_point = max_line.generate_mid_point_in(max_h=max_h, max_w=max_w)

        return center_point

    def generate_points_on_polygon(self, max_h=None, max_w=None):
        if max_h is None:
            max_h = self._shape.H
        if max_w is None:
            max_w = self._shape.W

        points = []
        for line in self.lines:
            points.extend(line.generate_points_on_line(max_h=max_h, max_w=max_w))
        return points

    def condense_lines(self, min_length, caller=None):

        old_len = -1
        while old_len != len(self.lines):
            i = 0
            bounds = [0, len(self.lines)]
            new_lines = []
            old_len = len(self.lines)

            while bounds[0] <= i < bounds[1]:
                # print(f" ----> {i}. line")
                line = self.lines[i]
                if line.length() < min_length:

                    if len(new_lines) > 0:
                        prev_line = new_lines[-1]
                    else:
                        prev_line = self.lines[i - 1 if i > 0 else len(self.lines) - 1]

                    if (i + 1) == bounds[1] and len(new_lines) > 0:
                        # print(f"{i + 1} < {bounds[1]}, L: {len(self.lines)}")
                        next_line = new_lines[0]
                    else:
                        next_line = self.lines[i + 1 if i + 1 < old_len else 0]

                    if next_line.similarity(line) >= prev_line.similarity(line):

                        self._debugger.resolve_print("PRINT_ATTACHED_LINES", "NEXT")

                        line.attach(next_line)
                        if (i + 1) == bounds[1] and len(new_lines) > 0:
                            new_lines[0] = line
                        else:
                            new_lines.append(line)
                        i += 2
                        bounds[0] += 2
                    else:
                        self._debugger.resolve_print("PRINT_ATTACHED_LINES", "PREV")
                        prev_line.attach(line)
                        if len(new_lines) == 0:
                            line = prev_line
                            bounds[1] -= 1
                            new_lines.append(line)
                        bounds[0] += 1
                        i += 1
                else:
                    bounds[0] += 1
                    i += 1
                    new_lines.append(line)
            self._debugger.resolve_print("PRINT_CONDENSED_LINES", f"Removed {len(self.lines) - len(new_lines)} lines")

            self.lines = new_lines
            if caller is not None:
                caller.call()

        def condensed(lines):
            def f():
                print("+++++++++++++++ CONDENSED LINES +++++++++++++++")
                for l in lines:
                    print("\t" + str(l))

            return f

        self._debugger.resolve_func("PRINT_ALL_CONDENSED_LINES", condensed(self.lines))

        self.__calc_params()


class Segment(Debuggable, ShapeDependable):
    def __init__(self, polygons: [Polygon], segment_class, main_polygon_index=None):
        super().__init__()
        self.segment_class = segment_class
        self.polygons: [Polygon] = polygons
        self.__main_polygon_index = main_polygon_index

    def apply_debugger(self, debugger: SegmentsDebugger):
        super(Segment, self).apply_debugger(debugger)
        for poly in self.polygons:
            poly.apply_debugger(debugger)

    def apply_shape(self, shape: Shape):
        super(Segment, self).apply_shape(shape)
        for poly in self.polygons:
            poly.apply_shape(shape)

    @property
    def main_polygon_index(self):
        return self.__main_polygon_index

    def get_main_polygon(self) -> Polygon:
        return self.polygons[self.__main_polygon_index]

    def condense_polygons(self, min_length=2):
        for polygon in self.polygons:
            polygon.condense_lines(min_length)


class ImageSegments(Debuggable, ShapeDependable):
    def __init__(self, segments: [Segment], classes, dim, debugger=None):
        super().__init__()
        self.__segments = segments
        self.__classes = classes
        self.__debugger = debugger
        self.__dim = dim
        if debugger is not None:
            self.apply_debugger(debugger)
        self.apply_shape(self.__dim)

    def apply_debugger(self, debugger: SegmentsDebugger):
        super(ImageSegments, self).apply_debugger(debugger)
        for segment in self.__segments:
            segment.apply_debugger(debugger)

    def apply_shape(self, shape: Shape):
        super(ImageSegments, self).apply_shape(shape)
        for segment in self.__segments:
            segment.apply_shape(shape)

    @property
    def segments(self):
        return self.__segments

    @property
    def classes(self):
        return self.__classes

    def scale_up(self, scale: float = 1.0):
        if abs(scale - 1.0) < 0.001:
            return
        for segment in self.segments:
            for polygon in segment.polygons:
                polygon.scale(scale)
        self._shape.scale(scale)

    @property
    def w(self):
        return self._shape.W

    @property
    def h(self):
        return self._shape.H


def derive_image_segments(json_path, class_map, w, h, debugger=None):
    with open(json_path, "r") as f:
        json_string = "".join(f.readlines())
        json_object = json.loads(json_string)

    labels = json_object["labels"]

    classes = set()
    classes.add(0)
    segments = []
    for j, label in enumerate(labels):
        json_class = label["label_class"]

        class_id = class_map[json_class]

        classes.add(class_id)

        regions = label["regions"]

        polygons = []

        min_y = np.inf
        min_x = np.inf
        max_y = -1
        max_x = -1
        main_polygon_index = -1

        for i, region in enumerate(regions):
            min_index = None
            min_value = w + 1

            local_min_y = np.inf
            local_min_x = np.inf
            local_max_y = -1
            local_max_x = -1
            for j, coord in enumerate(region):
                """
                    Koordinate se ukrstaju zbog drugacijeg formata PIL.Image klase
                """
                y = coord["x"]
                x = coord["y"]

                if y < min_value:
                    min_value = y
                    min_index = j
                if x > local_max_x:
                    local_max_x = x
                if y > local_max_y:
                    local_max_y = y
                if x < local_min_x:
                    local_min_x = x
                if y < local_min_y:
                    local_min_y = y

            if local_max_x > max_x and local_max_y > max_y and local_min_x < min_x and local_min_y < min_y:
                main_polygon_index = i
                min_y = local_min_y
                min_x = local_min_x
                max_y = local_max_y
                max_x = local_max_x

            direction = 1

            next_index = gen_next_index(min_index, region, direction)
            prev_index = gen_next_index(min_index, region, -direction)

            line, direction = generate_first_line(min_index, next_index, prev_index, region)

            next_index = gen_next_index(min_index, region, direction)

            lines: [Line] = [line]

            while next_index != min_index:
                last_index = gen_next_index(next_index, region, direction)

                prev_line = lines[-1]

                new_line = generate_line(next_index, last_index, region)

                pline1 = prev_line.pline
                pline2 = new_line.pline

                inv = is_inverting_line(pline1, pline2)

                if inv:
                    new_line.upper = not prev_line.upper
                else:
                    new_line.upper = prev_line.upper

                lines.append(new_line)

                next_index = last_index
            # proba

            polygon = Polygon(lines)
            # inspector = Inspector(polygon)
            inspector = None
            polygons.append(polygon)

        main_polygon = polygons[main_polygon_index]
        main_polygon.main = True
        length = 0
        max_index = -1
        for i, line in enumerate(main_polygon.lines):
            llength = line.length()
            if llength > length:
                max_index = i
                length = llength
        main_polygon.max_index = max_index

        for i, polygon in enumerate(polygons):
            if i == main_polygon_index:
                continue
            polygon.invert()
        segment = Segment(polygons, class_id, main_polygon_index)

        segments.append(segment)

    dim = Shape(w, h)
    mask = ImageSegments(segments, classes, dim, debugger)

    return mask


def derive_masks(image_segments: ImageSegments, class_id_to_value, dim=None, scale=2.0):
    h, w = image_segments.h, image_segments.w
    image_segments.scale_up(scale=scale)

    classes = np.array(list(image_segments.classes))

    if dim is None:
        out = np.zeros((h, w))
    else:
        out = np.zeros((h, w, dim))

    LINE_VAL = 33

    COL_VAL = 66

    MID_VAL = 100

    H_s, W_s = image_segments.h, image_segments.w

    pixel_map = np.zeros((H_s, W_s, 1))
    class_map = np.zeros((H_s, W_s))
    i = -1
    for segment in image_segments.segments:
        i += 1
        class_id = segment.segment_class

        polygons = segment.polygons

        mid_points = []

        for polygon in polygons:

            points_on_line = polygon.generate_points_on_polygon(max_h=H_s, max_w=W_s)

            center_point = polygon.generate_center_point(max_h=H_s, max_w=W_s)

            for point in points_on_line:
                pixel_map[point.x, point.y] = LINE_VAL
                class_map[point.x, point.y] = class_id

            pixel_map[center_point.x, center_point.y] = MID_VAL
            class_map[center_point.x, center_point.y] = class_id

            mid_points.append(center_point)

        color_left = True
        checked_mid_points = []
        for point in mid_points:
            if pixel_map[point.x, point.y] == MID_VAL:
                checked_mid_points.append(point)

        old_points = checked_mid_points

        while color_left:
            new_points = []
            for point in old_points:
                for i in range(4):
                    if i % 2 == 0:
                        x = point.x - 1 + i
                        y = point.y
                    else:
                        y = point.y - 2 + i
                        x = point.x
                    if not (H_s > x >= 0 and W_s > y >= 0):
                        continue

                    if pixel_map[x, y] == 0:
                        new_points.append(Point(x, y))
                        pixel_map[x, y] = COL_VAL
                        class_map[x, y] = class_id
            if len(new_points) == 0:
                color_left = False
            del old_points
            gc.collect()
            old_points = new_points

    class_map = scale_img(class_map, h, w)

    for i in range(h):
        for j in range(w):
            out[i, j] = class_id_to_value(int(class_map[i][j]))

    return out, classes
