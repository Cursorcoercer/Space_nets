import parameters as params
import numpy as np
from scipy import spatial
import pyglet
import colorsys
import random
import math
import time
import os


def dist_sq(coords_1, coords_2):
    # find the distance squared between two coordinates
    dist = 0
    for f in range(len(coords_1)):
        dist += (coords_1[f] - coords_2[f])**2
    return dist


def not_part_of_triplet(lis, elem):
    for other in lis:
        if other == elem:
            continue
        if elem[0] in other:
            for other_2 in lis:
                if other_2 == elem or other_2 == other:
                    continue
                if ((other_2[0] in elem or other_2[0] in other) and
                        (other_2[1] in elem or other_2[1] in other)):
                    return False
    return True


def z_cross(vector1, vector2):
    # return the z component of a cross product
    return vector1[0] * vector2[1] - vector1[1] * vector2[0]


def same_sign_cross(vector, bound1, bound2):
    # determine if two vectors produce the same sign on the z component of a 3d cross product
    # in lay terms check if vector lies between bounds (sort of)
    return 0 > z_cross(vector, bound1) * z_cross(vector, bound2)


def is_color(lis):
    if len(lis) != 3:
        return False
    for f in lis:
        if type(f) != int:
            return False
        elif f < 0 or 255 < f:
            return False
    return True


def get_new_file_name(path):
    n = 1
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = time_stamp + ".png"
    while os.path.exists(os.path.join(path, file_name)):
        n += 1
        file_name = time_stamp + "%s.png" % n
    return os.path.join(path, file_name)


def chunks(lst, n):
    # Yield successive n-sized chunks from lst
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def stutter(lis, n):
    # repeat a list on a repetition of n
    new_lis = []
    for f in chunks(lis, n):
        new_lis += 2 * f
    return new_lis


def random_color():
    # return a random color
    return random.randrange(256), random.randrange(256), random.randrange(256)


def color_avg(colors):
    # return an average of the colors given
    if not colors:
        # return a random color if no colors are passed
        return random_color()
    f_len = len(colors[0])
    avg = f_len * [0]
    for f in range(f_len):
        for color in colors:
            avg[f] += color[f]
        avg[f] = int(avg[f] / len(colors))
    return tuple(avg)


def step(r, g, b, repetitions=8):
    lum = math.sqrt(0.241 * r + 0.691 * g + 0.068 * b)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)
    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum2 = repetitions - lum2
    return h2, lum2, v2


def k_means(path, k):
    # return k mean colors from an image
    form = "RGB"
    f_len = len(form)
    if not os.path.exists(path):
        raise ValueError("Invalid image path")
    image = pyglet.image.load(path)
    pitch = image.width * f_len
    pixels = list(tuple(c) for c in chunks(image.get_data(form, pitch), f_len))
    # a couple of preliminary checks for special cases
    if k == len(pixels):
        # the number given is the number of pixels
        return pixels
    pixel_set = list(set(pixels))
    if k == len(pixel_set):
        # k is equal to the number of unique pixels
        return pixel_set
    # do it the hard way
    # but first let's parse down our pixel list so as to be practical
    pixels.sort(key=lambda x: step(x[0], x[1], x[2]))
    # the smaller the chunks the better the k means accuracy at the expense of time
    chunk_size = len(pixels) // 1000
    new_pixels = []
    for f in range(0, len(pixels), chunk_size):
        new_pixels.append(color_avg(pixels[f:f + chunk_size]))
    clusters = len(new_pixels) * [0]
    old_means = []
    means = []
    for f in range(k):
        old_means.append((-1, -1, -1))
        means.append(random_color())
    while old_means != means:
        old_means = list(means)
        # group the clusters
        for f in range(len(clusters)):
            clusters[f] = min(range(k), key=lambda x: dist_sq(means[x], new_pixels[f]))
        # adjust the means
        for f in range(k):
            means[f] = color_avg(list(new_pixels[g] for g in range(len(new_pixels)) if clusters[g] == f))
    return means


class Simplex:

    def __init__(self, scale=1, seed=0):
        self.f2 = (math.sqrt(3) - 1) / 2
        self.g2 = (3 - math.sqrt(3)) / 6
        self.r = 0.6  # r value for the simplex
        self.scale = scale
        self.grad_num = 12
        self.grad = []
        for f in range(self.grad_num):
            angle = f * math.tau / self.grad_num
            self.grad.append([math.cos(angle), math.sin(angle)])

    def get_noise(self, x, y, num=1, seed=0):
        # return simplex noise
        x = x / self.scale
        y = y / self.scale
        total_noise = np.zeros(num)
        # Skew the input space to determine which simplex cell we're in
        s = (x + y) * self.f2
        i = int(x + s)
        j = int(y + s)
        # Unskew the cell origin back to (x,y) space
        t = (i + j) * self.g2
        x0 = i - t
        y0 = j - t
        # The x,y distances from the cell origin
        xd = x - x0
        yd = y - y0
        # Determine which simplex we are in.
        i1, j1 = 0, 0  # Offsets for second (middle) corner of simplex in (i,j) coords
        if xd > yd:  # lower triangle, XY order: (0,0)->(1,0)->(1,1)
            i1 = 1
            j1 = 0
        else:  # upper triangle, YX order: (0,0)->(0,1)->(1,1)
            i1 = 0
            j1 = 1
        # A step of (1,0) in (i,j) means a step of (1-g,-g) in (x,y), and
        # a step of (0,1) in (i,j) means a step of (-g,1-g) in (x,y), where
        x1 = xd - i1 + self.g2  # Offsets for middle corner in (x,y) unskewed coords
        y1 = yd - j1 + self.g2
        x2 = xd - 1.0 + 2.0 * self.g2  # Offsets for last corner in (x,y) unskewed coords
        y2 = yd - 1.0 + 2.0 * self.g2
        # Calculate the contribution from the three corners
        t0 = 0.5 - xd * xd - yd * yd
        t1 = 0.5 - x1 * x1 - y1 * y1
        t2 = 0.5 - x2 * x2 - y2 * y2
        if t0 > 0:
            t0 *= t0
            for f in range(num):
                gi0 = hash((i, j, seed+f)) % self.grad_num
                total_noise[f] += t0 * t0 * (self.grad[gi0][0] * xd + self.grad[gi0][1] * yd)
        if t1 > 0:
            t1 *= t1
            for f in range(num):
                gi1 = hash((i + i1, j + j1, seed+f)) % self.grad_num
                total_noise[f] += t1 * t1 * (self.grad[gi1][0] * x1 + self.grad[gi1][1] * y1)
        if t2 > 0:
            t2 *= t2
            for f in range(num):
                gi2 = hash((i + 1, j + 1, seed+f)) % self.grad_num
                total_noise[f] += t2 * t2 * (self.grad[gi2][0] * x2 + self.grad[gi2][1] * y2)
        # Add contributions from each corner to get the final noise value.
        # The result is scaled to return values in the interval [-1,1].
        return 101 * total_noise


class Color:

    def __init__(self, color, scsz):
        self.data = color
        self.scsz = scsz
        self.seed = random.random()
        self.image = None
        self.pixels = None
        self.color_format = 'RGB'
        self.f_len = len(self.color_format)
        self.color_type = self.data_type()
        self.noise_vals = (params.white_noise_strength, params.red_noise_strength,
                           params.green_noise_strength, params.blue_noise_strength)
        self.noise_num = sum(bool(f) for f in self.noise_vals)
        if self.noise_num:
            self.noise = Simplex(scale=self.scsz[1]/params.noise_density)

    def data_type(self):
        if self.data is None:
            return None  # random colors
        if type(self.data) == str:
            if not os.path.exists(self.data):
                raise ValueError("Invalid image path")
            self.image = pyglet.image.load(self.data)
            pitch = self.image.width * self.f_len
            self.pixels = self.image.get_data(self.color_format, pitch)
            return 2  # colors from an image
        if len(self.data) == 2:
            if type(self.data[0]) == int and type(self.data[1]) == str:
                self.data = self.data[::-1]
            if type(self.data[0]) == str or type(self.data[1]) == int:
                self.data = k_means(*self.data)
                return 1  # we turned an image into a color palette
        if is_color(self.data):
            return 0  # single color
        for c in self.data:
            if not is_color(c):
                raise ValueError("Invalid color value")
        return 1  # color palette

    def get_pixel(self, x, y):
        pos = self.image.width * y * self.f_len + x * self.f_len
        return self.pixels[pos:pos + self.f_len]

    def add_noise(self, color, coords):
        new_color = list(color)
        noises = self.noise.get_noise(coords[0], coords[1], num=self.noise_num, seed=self.seed)
        used = int(bool(self.noise_vals[0]))
        for f in range(3):
            if self.noise_vals[0]:
                # if white, it gets first noise
                new_color[f] += noises[0] * self.noise_vals[0]
            if self.noise_vals[f+1]:
                new_color[f] += noises[used] * self.noise_vals[f+1]
                used += 1
            new_color[f] = min(max(int(new_color[f]), 0), 255)
        return new_color

    def reseed(self):
        self.seed = random.random()

    def get_color(self, element, coords, seed_add=0):
        color_pick = []
        if self.color_type is None:  # random color
            t_rand = hash((element, self.seed+seed_add))
            color_pick = t_rand % 256, (t_rand//256) % 256, ((t_rand//256)//256) % 256
        elif self.color_type == 0:  # single color
            color_pick = self.data
        elif self.color_type == 1:  # color palette
            t_rand = hash((element, self.seed+seed_add))
            color_pick = self.data[t_rand % len(self.data)]
        elif self.color_type == 2:  # color map
            # for this the element should be point coords
            # indicating where on the map to draw the color from
            x = int(coords[0]/self.scsz[0] * (self.image.width - 1))
            y = int(coords[1]/self.scsz[1] * (self.image.height - 1))
            color_pick = self.get_pixel(x, y)
        if any(self.noise_vals):
            color_pick = self.add_noise(color_pick, coords)
        return color_pick


class Field:

    def __init__(self, point_num, height, feel_num, air_resistance, bounciness, funcs, screen_size,
                 symmetric=True, init_vel=0):
        self.num = point_num
        self.height = height
        self.feel_num = feel_num
        self.air = air_resistance
        self.bounce = bounciness
        self.funcs = funcs
        if type(self.funcs) != tuple:
            self.funcs = (self.funcs,)
        self.scsz = screen_size
        self.symmetric_forces = symmetric
        self.init_vel = init_vel
        self.aspect_ratio = self.scsz[0]/self.scsz[1]  # aspect ratio of screen
        self.real_ratio = self.scsz[1]/self.height  # ratio of actual height to screen height
        self.update_num = 0
        self.positions = np.zeros((self.num, 2))
        self.velocities = np.zeros((self.num, 2))
        self.kd_tree = spatial.cKDTree(self.positions)
        self.lines = np.zeros((self.num, self.feel_num))
        self.sym_lines = [set() for f in range(self.num)]
        self.line_set = set()
        self.rect_lines = bool(params.line_width > 10)
        if self.rect_lines:
            self.line_type = pyglet.gl.GL_QUADS
        else:
            self.line_type = pyglet.gl.GL_LINES
        self.triangles = []
        self.p_data = []
        self.pc_data = []
        self.l_data = []
        self.lc_data = []
        self.tr_data = []
        self.trc_data = []
        self.color_fade = False
        self.triangle_show = False
        self.line_show = False
        self.point_show = True
        self.delaunay_tri = False
        self.grabbed = None
        self.stasis = set()
        self.last_pos = (0, 0)
        self.point_color = Color(params.point_color, self.scsz)
        self.line_color = Color(params.line_color, self.scsz)
        self.triangle_color = Color(params.triangle_color, self.scsz)
        self.reset()

    def reset(self):
        # generates all particles within the field of the screen
        # origin as bottom left and height as coordinate of top
        for f in range(self.num):
            if f in self.stasis:
                continue
            self.positions[f] = [self.height * self.aspect_ratio * random.random(), self.height * random.random()]
            if self.init_vel:
                # generates a random angle for the velocity to be at
                rand_angle = math.tau * random.random()
                self.velocities[f] = [self.init_vel * math.cos(rand_angle), self.init_vel * math.sin(rand_angle)]
        # get the lines right immediately
        self.sym_lines = [set() for f in range(self.num)]
        self.line_set = set()
        for f in range(self.num):
            closest = self.find_closest(f)
            self.lines[f] = closest[1]
            self.sym_lines[f].update(closest[1])
            for c in closest[1]:
                c = int(c)
                self.sym_lines[c].add(f)
                self.line_set.add(tuple(sorted((f, c))))
        self.triangles = self.find_triangles()
        self.prepare_data()

    def set_point_show(self, val=None):
        if val is not None:
            self.point_show = val
        else:
            # default to toggle
            self.point_show = not self.point_show
        if self.point_show:
            self.prepare_data()

    def set_line_show(self, val=None):
        if val is not None:
            self.line_show = val
        else:
            # default to toggle
            self.line_show = not self.line_show
        if self.line_show:
            self.prepare_data()

    def set_tri_show(self, val=None):
        if val is not None:
            self.triangle_show = val
        else:
            # default to toggle
            self.triangle_show = not self.triangle_show
        if self.triangle_show:
            self.triangles = self.find_triangles()
            self.prepare_data()

    def set_del(self, val=None):
        if val is not None:
            self.delaunay_tri = val
        else:
            # default to toggle
            self.delaunay_tri = not self.delaunay_tri
        self.triangles = self.find_triangles()
        self.prepare_data()

    def set_fade(self, val=None):
        if val is not None:
            self.color_fade = val
        else:
            # default to toggle
            self.color_fade = not self.color_fade
        self.prepare_data()

    def set_grabbed(self, coords):
        if coords is None:
            self.grabbed = None
        else:
            real_coords = (coords[0] / self.real_ratio, coords[1] / self.real_ratio)
            self.grabbed = self.kd_tree.query(real_coords)[1]

    def move_grabbed(self, position=None, velocity=None):
        if self.grabbed is None:
            return None
        if position is not None:
            self.last_pos = np.array(position)
        self.positions[self.grabbed] = self.last_pos / self.real_ratio
        if any(self.out_of_bounds(self.positions[self.grabbed])):
            self.handle_out_of_bounds_point(self.grabbed)
        if velocity is not None:
            self.velocities[self.grabbed] = np.array(velocity) / self.real_ratio

    def stasis_grab(self):
        # put the grabbed point into stasis if there is one
        if self.grabbed is None:
            # if none grabbed, clear stasis
            self.stasis = set()
        else:
            self.stasis.add(self.grabbed)
            self.set_grabbed(None)

    def out_of_bounds(self, position):
        # return whether a point is out of bounds or not
        return (position[0] < 0 or self.height * self.aspect_ratio < position[0],
                position[1] < 0 or self.height < position[1])

    def handle_out_of_bounds_point(self, point_num):
        self.velocities[point_num] = [0, 0]
        self.positions[point_num][0] = min(max(0, self.positions[point_num][0]), self.aspect_ratio * self.height)
        self.positions[point_num][1] = min(max(0, self.positions[point_num][1]), self.height)

    def find_closest(self, num):
        plus_one = self.kd_tree.query(self.positions[num], k=self.feel_num+1)
        for f in range(len(plus_one[1])):
            if num == plus_one[1][f]:
                return np.delete(plus_one, f, 1)
        return np.delete(plus_one, -1, 1)

    def move_point(self, point_num, extra=np.zeros(2), div=1):
        return self.positions[point_num] + (self.velocities[point_num] / div) + extra

    def update_vel(self):
        # update the velocities of all the points
        # forces from other points
        self.sym_lines = [set() for f in range(self.num)]
        self.line_set = set()
        for f in range(self.num):
            closest = self.find_closest(f)
            self.lines[f] = closest[1]
            self.sym_lines[f].update(closest[1])
            stasis_flag = bool(f in self.stasis)
            if stasis_flag and not self.symmetric_forces:
                continue
            for c in range(self.feel_num):
                p = int(closest[1][c])
                act_dist = closest[0][c]
                self.sym_lines[p].add(f)
                self.line_set.add(tuple(sorted((f, p))))
                if act_dist == 0:
                    # points are on top of each other
                    continue
                force = self.funcs[f % len(self.funcs)](act_dist)
                if not stasis_flag:
                    self.velocities[f] += force * (self.positions[f] - self.positions[p]) / act_dist
                if self.symmetric_forces and p not in self.stasis:
                    # this makes forces symmetric, but obeying newton is for nerds
                    self.velocities[p] -= force * (self.positions[f] - self.positions[p]) / act_dist
        # forces from the bounds
        for f in range(self.num):
            out_x, out_y = self.out_of_bounds(self.move_point(f))
            if out_x:
                self.velocities[f][0] *= -1
                self.velocities[f] *= self.bounce
            if out_y:
                self.velocities[f][1] *= -1
                self.velocities[f] *= self.bounce
            # then we account for air resistance
            self.velocities[f] *= self.air

    def not_obstructed(self, points):
        # determines whether a convex polygon is obstructed by any line
        for p in range(len(points)):
            o1 = (p - 1) % len(points)
            o2 = (p + 1) % len(points)
            v1 = self.positions[points[o1]] - self.positions[points[p]]
            v2 = self.positions[points[o2]] - self.positions[points[p]]
            for f in self.sym_lines[points[p]]:
                f = int(f)
                if f in points:
                    if len(points) > 3 and (points.index(f) not in (o1, o2)):
                        return False
                    else:
                        continue
                to_test = self.positions[f] - self.positions[points[p]]
                if same_sign_cross(to_test, v1, v2):
                    if not same_sign_cross(v2, v1, to_test):
                        return False
        return True

    def find_triangles(self):
        # return a list of all triangle tuples
        if self.delaunay_tri:
            good_tri = []
            del_tris = spatial.Delaunay(self.positions)
            for tri in del_tris.simplices:
                good_tri.append(tuple(sorted(tri)))
            return good_tri
        else:
            all_tri = set()
            good_tri = []
            for p in range(self.num):
                # neighbors = set(f for f in range(self.num) if (p in self.lines[f] or f in self.lines[p]))
                for c in self.lines[p]:
                    for c2 in self.lines[int(c)]:
                        if c2 in self.lines[p]:
                            all_tri.add(tuple(sorted((p, int(c), int(c2)))))
            for tri in all_tri:
                if self.not_obstructed(tri):
                    good_tri.append(tri)
            return good_tri

    def reseed(self):
        self.point_color.seed = random.random()
        self.line_color.seed = random.random()
        self.triangle_color.seed = random.random()
        self.prepare_data()

    def update(self, global_move=np.zeros(2), num=1):
        # first we update the velocities of the points based on their forces
        self.update_num = (self.update_num + 1) % num
        if self.update_num == 1 or num == 1:
            self.update_vel()
            if self.triangle_show:
                self.triangles = self.find_triangles()
        for f in range(self.num):
            # then we update the position of the points based on their velocities
            if f in self.stasis:
                continue
            self.positions[f] = self.move_point(f, extra=global_move, div=num)
            # then we account for any points that somehow slipped out of bounds
            if any(self.out_of_bounds(self.positions[f])):
                self.handle_out_of_bounds_point(f)
        # then we update the p_data for the vertex buffer
        self.move_grabbed()
        self.prepare_data()

    def resize(self, screen_size):
        self.scsz = screen_size
        self.aspect_ratio = self.scsz[0]/self.scsz[1]  # aspect ratio of screen
        self.real_ratio = self.scsz[1]/self.height  # ratio of actual height to screen height
        self.update()

    def avg_points(self, points):
        # converts a flat list of points to their average
        avg_x = 0
        avg_y = 0
        for f in points[::2]:
            avg_x += f
        for f in points[1::2]:
            avg_y += f
        num = len(points) // 2
        return avg_x / num, avg_y / num

    def points_to_rect(self, point_nums):
        # converts a list of point indices to a rectangle that acts as a line
        data = []
        point1 = self.positions[point_nums[0]] * self.real_ratio
        point2 = self.positions[point_nums[1]] * self.real_ratio
        size = math.sqrt(dist_sq(point1, point2))
        offset = np.array([point2[1] - point1[1], point1[0] - point2[0]])
        offset *= (params.line_width / 2) / size
        data += list(point1 + offset)
        data += list(point1 - offset)
        data += list(point2 - offset)
        data += list(point2 + offset)
        return data

    def points_to_real(self, point_nums):
        # converts a list of point indices to their corresponding real coords as a flattened list
        data = []
        for num in point_nums:
            data += list(self.positions[int(num)] * self.real_ratio)
        return data

    def prepare_data(self):
        # transforms p_data into screen coordinates
        # then puts it in proper openGL type
        self.p_data = []
        self.pc_data = []
        self.l_data = []
        self.lc_data = []
        self.tr_data = []
        self.trc_data = []
        if self.point_show:
            for f in range(self.num):
                self.p_data += self.points_to_real((f,))
                self.pc_data += self.point_color.get_color(f, self.p_data[-2:])
        if self.line_show:
            for li in self.line_set:
                if self.rect_lines:
                    self.l_data += self.points_to_rect(li)
                    last_points = self.points_to_real(li)
                else:
                    self.l_data += self.points_to_real(li)
                    last_points = self.l_data[-4:]
                if self.color_fade:
                    self.lc_data += self.line_color.get_color(li, last_points[:2])
                    self.lc_data += self.line_color.get_color(li, last_points[-2:], seed_add=1)
                else:
                    self.lc_data += 2 * self.line_color.get_color(li, self.avg_points(last_points))
            if self.rect_lines:
                self.lc_data = stutter(self.lc_data, 3)
        if self.triangle_show:
            for tri in self.triangles:
                self.tr_data += self.points_to_real(tri)
                if self.color_fade:
                    for t in range(len(tri)):
                        self.trc_data += self.triangle_color.get_color(tri, (self.tr_data[t*2-6],
                                                                             self.tr_data[t*2-5]), seed_add=t)
                else:
                    self.trc_data += 3 * self.triangle_color.get_color(tri, self.avg_points(self.tr_data[-6:]))

    def draw(self):
        if self.triangle_show:
            pyglet.graphics.draw(len(self.tr_data)//2, pyglet.gl.GL_TRIANGLES,
                                 ('v2f', self.tr_data), ('c3B', self.trc_data))
        if self.line_show:
            pyglet.graphics.draw(len(self.l_data)//2, self.line_type,
                                 ('v2f', self.l_data), ('c3B', self.lc_data))
        if self.point_show:
            pyglet.graphics.draw(len(self.positions), pyglet.gl.GL_POINTS,
                                 ('v2f', self.p_data), ('c3B', self.pc_data))


class GUI(pyglet.window.Window):

    def __init__(self):
        title = 'point interactions'
        if params.anti_aliasing:
            config = pyglet.gl.Config(sample_buffers=1, samples=4)
        else:
            config = None
        super(GUI, self).__init__(caption=title, config=config, fullscreen=True, resizable=True)
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.set_minimum_size(100, 100)
        pyglet.gl.glReadBuffer(pyglet.gl.GL_FRONT)
        pyglet.gl.glDrawBuffer(pyglet.gl.GL_BACK)

        # set parameters
        if params.behavior_preset == "mesh":
            self.dots = Field(100, 100, 2, 0.9, 1, lambda x: 1/3/x, self.get_size())
        elif params.behavior_preset == "blob":
            self.dots = Field(100, 100, 4, 0.9, 1, lambda x: 2-x/5, self.get_size())
        elif params.behavior_preset == "shrink":
            self.dots = Field(100, 10, 5, 0.9, 1, lambda x: -math.atan(x)/10, self.get_size())
        elif params.behavior_preset == "fireworks":
            self.dots = Field(100, 100, 2, 0.9, 1, lambda x: -1/x/50, self.get_size())
        elif params.behavior_preset == "amoeba":
            self.dots = Field(100, 100, 4, 0.9, 1, (lambda x: 1.2-x/5, lambda x: 2.4-x/5),
                              self.get_size(), symmetric=False)
        elif params.behavior_preset == "gas":
            self.dots = Field(100, 100, 4, 1, 1, lambda x: 0, self.get_size(), init_vel=1)
        else:
            self.dots = Field(params.number_of_points, 100, params.point_connections,
                              params.air_resistance, params.bounciness, params.point_function,
                              self.get_size(), symmetric=params.symmetric_forces,
                              init_vel=params.init_velocity)

        pyglet.gl.glClearColor(params.background_color[0]/255, params.background_color[1]/255,
                               params.background_color[2]/255, 1.0)
        pyglet.gl.glPointSize(params.point_size)
        pyglet.gl.glLineWidth(params.line_width)

        mouse_map = {"left": pyglet.window.mouse.LEFT, "right": pyglet.window.mouse.RIGHT,
                     "middle": pyglet.window.mouse.MIDDLE}
        self.active_button = mouse_map[params.mouse_button.lower()]
        self.keys = {}
        self.full = True
        self.pause = False
        self.stain = False
        self.fps_show = False
        self.slow_down = 1
        self.speed_num = 0
        self.speed = params.manual_speeds[self.speed_num]

    def on_key_press(self, symbol, modifiers):
        self.keys[pyglet.window.key.symbol_string(symbol)] = True
        key_str = pyglet.window.key.symbol_string(symbol)
        if key_str == params.fullscreen_key:  # toggle fullscreen
            self.full = not self.full
            self.set_fullscreen(self.full)
        elif key_str == params.pause_key:  # pause
            self.pause = not self.pause
        elif key_str == params.force_frame_key:  # go forward one frame
            self.update_dots()
        elif key_str == params.stain_key:  # toggle stain
            self.stain = not self.stain
        elif key_str == params.fps_key:  # toggle fps reading
            self.fps_show = not self.fps_show
        elif key_str == params.reset_key:  # reset field
            self.dots.reset()
        elif key_str == params.slow_key:  # make the animation go slow
            self.slow_down = params.slow_down - self.slow_down
        elif key_str == params.point_key:  # show points
            self.dots.set_point_show()
        elif key_str == params.line_key:  # show lines
            self.dots.set_line_show()
        elif key_str == params.triangle_key:  # show triangles
            self.dots.set_tri_show()
        elif key_str == params.triangle_type_key:  # change triangle type
            self.dots.set_del()
        elif key_str == params.reseed_key:  # re-seed the random colors
            self.dots.reseed()
        elif key_str == params.color_fade_key:  # toggle fade
            self.dots.set_fade()
        elif key_str == params.stasis_key:
            self.dots.stasis_grab()
        elif key_str == params.capture_key:  # capture the image on screen into a file
            pyglet.image.get_buffer_manager().get_color_buffer().save(get_new_file_name(params.path_name))
            # set the buffers again so that stain continues to work
            pyglet.gl.glReadBuffer(pyglet.gl.GL_FRONT)
            pyglet.gl.glDrawBuffer(pyglet.gl.GL_BACK)
        elif key_str == params.speed_toggle:  # toggle which speed manual movement uses
            self.speed_num += 1
            self.speed_num %= len(params.manual_speeds)
            self.speed = params.manual_speeds[self.speed_num]
            print(self.speed)
        elif key_str == params.quit_key:  # exit
            self.close()

    def on_key_release(self, symbol, modifiers):
        self.keys[pyglet.window.key.symbol_string(symbol)] = False

    def is_pressed(self, key_str):
        return bool((key_str in self.keys) and (self.keys[key_str]))

    def on_mouse_press(self, x, y, button, modifiers):
        if button == self.active_button:
            self.dots.set_grabbed((x, y))
            self.dots.move_grabbed((x, y))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.dots.move_grabbed((x, y), velocity=(dx, dy))

    def on_mouse_release(self, x, y, button, modifiers):
        if button == self.active_button:
            self.dots.set_grabbed(None)

    def on_resize(self, width, height):
        self._projection.set(width, height, *self.get_framebuffer_size())
        self.dots.resize((width, height))

    def update_dots(self):
        extra_move = np.zeros(2)
        extra_move += [self.is_pressed(params.right_key), self.is_pressed(params.up_key)]
        extra_move -= [self.is_pressed(params.left_key), self.is_pressed(params.down_key)]
        extra_move *= self.speed
        self.dots.update(global_move=extra_move, num=self.slow_down)

    def update(self, dt):
        if not self.pause:
            self.update_dots()
        if not self.stain:
            pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
        else:
            # copy front buffer to back buffer for reliable staining
            pyglet.gl.glCopyPixels(0, 0, self.width, self.height, pyglet.gl.GL_COLOR)
        self.dots.draw()
        if self.fps_show:
            self.fps_display.draw()


if __name__ == '__main__':
    # just make double sure that there are no path issues
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(params.path_name):
        os.mkdir(params.path_name)

    # run the program
    window = GUI()
    pyglet.clock.schedule_interval(window.update, 1/params.FPS)
    pyglet.app.run()
