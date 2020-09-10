import parameters as params
import pyglet
from pyglet.window import key
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


def step(r, g, b, repetitions=1):
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
    pixels.sort(key=lambda x: step(x[0], x[1], x[2], params.pre_means_sorting_smoothness))
    chunk_size = len(pixels) // params.pre_means_clustering
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


class Point:

    def __init__(self, position, velocity):
        self.pos = position
        self.vel = velocity

    def move(self, div=1):
        self.pos[0] += self.vel[0] / div
        self.pos[1] += self.vel[1] / div

    def pseudo_move(self):
        # don't actually move, but return where you would move to
        return self.pos[0] + self.vel[0], self.pos[1] + self.vel[1]

    def distance(self, other):
        # find the distance between two points
        return dist_sq(self.pos, other.pos)


class Color:

    def __init__(self, color):
        self.data = color
        self.seed = random.random()
        self.image = None
        self.pixels = None
        self.color_format = 'RGB'
        self.f_len = len(self.color_format)
        self.color_type = self.data_type()

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

    def reseed(self):
        self.seed = random.random()

    def get_color(self, element, number=1, seed_add=0):
        if number > 1:
            # to get multiple different colors for the same element
            # do not use with type 2 data
            color_list = tuple()
            for f in range(int(number)):
                color_list += self.get_color(element, seed_add=f)
            return color_list
        if self.color_type is None:  # random color
            t_rand = hash((element, self.seed+seed_add))
            return t_rand % 256, (t_rand//256) % 256, ((t_rand//256)//256) % 256
        if self.color_type == 0:  # single color
            return self.data
        if self.color_type == 1:  # color palette
            t_rand = hash((element, self.seed+seed_add))
            return self.data[t_rand % len(self.data)]
        if self.color_type == 2:  # color map
            # for this the element should be a tuple of two floats from 0 to 1
            # indicating where on the map to draw the color from
            x = int(element[0] * (self.image.width - 1))
            y = int(element[1] * (self.image.height - 1))
            return self.get_pixel(x, y)


class Field:

    def __init__(self, point_num, height, feel_num, air_resistance, bounciness, funcs, screen_size,
                 symmetric=True, init_vel=0):
        self.num = point_num
        self.height = height
        self.feel_num = feel_num
        self.air = air_resistance
        self.bounce = bounciness
        self.funcs = funcs
        self.scsz = screen_size
        self.symmetric_forces = symmetric
        self.init_vel = init_vel
        self.aspect_ratio = self.scsz[0]/self.scsz[1]  # aspect ratio of screen
        self.real_ratio = self.scsz[1]/self.height  # ratio of actual height to screen height
        self.update_num = 0
        self.points = []
        self.lines = []
        self.triangles = []
        self.p_data = []
        self.pc_data = []
        self.l_data = []
        self.lc_data = []
        self.tr_data = []
        self.trc_data = []
        self.color_fade = False
        self.grabbed = None
        self.last_pos = (0, 0)
        self.point_color = Color(params.point_color)
        self.line_color = Color(params.line_color)
        self.triangle_color = Color(params.triangle_color)
        self.reset()

    def new_point(self):
        # generates a random new point
        if self.init_vel:
            # generates a random angle or the velocity to be at
            rand_angle = math.tau * random.random()
            rand_vel = [self.init_vel * math.cos(rand_angle), self.init_vel * math.sin(rand_angle)]
        else:
            rand_vel = [0, 0]
        rand_pos = [self.height * self.aspect_ratio * random.random(), self.height * random.random()]
        return Point(rand_pos, rand_vel)

    def reset(self):
        # generates all particles within the field of the screen
        # origin as bottom left and height as coordinate of top
        self.points = []
        for f in range(self.num):
            self.points.append(self.new_point())
            self.lines.append([])
        # get the lines right immediately
        for f in range(len(self.points)):
            closest = self.find_closest(f)
            self.lines[f] = []
            for c in closest:
                self.lines[f].append(c[0])
        self.triangles = self.find_triangles()
        self.prepare_data()

    def find_closest(self, num):
        closest = []
        for p in range(len(self.points)):
            if p == num:
                continue
            if len(closest) < self.feel_num:
                closest.append((p, self.points[p].distance(self.points[num])))
                if len(closest) == self.feel_num:
                    closest.sort(key=lambda x: x[1])
            else:
                temp_dist = self.points[p].distance(self.points[num])
                if temp_dist < closest[-1][1]:
                    closest[-1] = (p, temp_dist)
                    closest.sort(key=lambda x: x[1])
        return closest

    def closest_point(self, coords):
        return min(range(len(self.points)), key=lambda x: dist_sq(self.points[x].pos, coords))

    def set_grabbed(self, coords):
        if coords is None:
            self.grabbed = None
        else:
            real_coords = (coords[0] / self.real_ratio, coords[1] / self.real_ratio)
            self.grabbed = self.closest_point(real_coords)

    def move_grabbed(self, position=None, velocity=None):
        if self.grabbed is None:
            return None
        if position is not None:
            self.last_pos = position
        self.points[self.grabbed].pos[0] = self.last_pos[0] / self.real_ratio
        self.points[self.grabbed].pos[1] = self.last_pos[1] / self.real_ratio
        if self.out_of_bounds(self.points[self.grabbed].pos):
            self.handle_out_of_bounds_point(self.grabbed)
        if velocity is not None:
            self.points[self.grabbed].vel[0] = velocity[0] / self.real_ratio
            self.points[self.grabbed].vel[1] = velocity[1] / self.real_ratio

    def out_of_bounds(self, position):
        # return a boolean tuple with out of bounds for horizontal and vertical
        return (position[0] < 0 or self.height * self.aspect_ratio < position[0],
                position[1] < 0 or self.height < position[1])

    def handle_out_of_bounds_point(self, point_num):
        self.points[point_num].vel = [0, 0]
        self.points[point_num].pos[0] = min(max(0, self.points[point_num].pos[0]), self.aspect_ratio * self.height)
        self.points[point_num].pos[1] = min(max(0, self.points[point_num].pos[1]), self.height)

    def update_vel(self):
        # forces from other points
        for f in range(len(self.points)):
            closest = self.find_closest(f)
            self.lines[f] = []
            for c in closest:
                self.lines[f].append(c[0])
                act_dist = math.sqrt(c[1])
                if not act_dist:
                    # points are on top of each other
                    continue
                force = self.funcs[f % len(self.funcs)](act_dist)
                x_force = force * (self.points[f].pos[0] - self.points[c[0]].pos[0]) / act_dist
                y_force = force * (self.points[f].pos[1] - self.points[c[0]].pos[1]) / act_dist
                self.points[f].vel[0] += x_force
                self.points[f].vel[1] += y_force
                if self.symmetric_forces:
                    # this makes forces symmetric, but obeying newton is for nerds
                    self.points[c[0]].vel[0] -= x_force
                    self.points[c[0]].vel[1] -= y_force
        # forces from the bounds
        for f in range(len(self.points)):
            pot_pos = self.points[f].pseudo_move()
            out_x, out_y = self.out_of_bounds(pot_pos)
            if out_x:
                self.points[f].vel[0] *= -self.bounce
                self.points[f].vel[1] *= self.bounce
            if out_y:
                self.points[f].vel[0] *= self.bounce
                self.points[f].vel[1] *= -self.bounce
            # then we account for air resistance
            self.points[f].vel[0] *= self.air
            self.points[f].vel[1] *= self.air

    def find_triangles(self):
        all_tri = set()
        for p in range(len(self.points)):
            p_tri = set()
            neighbors = set(self.lines[p] + [f for f in range(len(self.points)) if p in self.lines[f]])
            for c in neighbors:
                for c2 in self.lines[c]:
                    if c2 in neighbors:
                        p_tri.add(tuple(sorted((c, c2))))
            for tri in p_tri:
                if not_part_of_triplet(p_tri, tri):
                    all_tri.add(tuple(sorted((p, tri[0], tri[1]))))
        return list(all_tri)

    def reseed(self):
        self.point_color.seed = random.random()
        self.line_color.seed = random.random()
        self.triangle_color.seed = random.random()

    def change_fade(self):
        self.color_fade = not self.color_fade

    def update(self, num=1):
        # first we update the velocities of the points based on their forces
        self.update_num = (self.update_num + 1) % num
        if self.update_num == 1 or num == 1:
            self.update_vel()
            self.triangles = self.find_triangles()
        for f in range(len(self.points)):
            # then we update the position of the points based on their velocities
            self.points[f].move(div=num)
            # then we account for any points that somehow slipped out of bounds
            if any(self.out_of_bounds(self.points[f].pos)):
                self.handle_out_of_bounds_point(f)
        # then we update the p_data for the vertex buffer
        self.move_grabbed()
        self.prepare_data()

    def resize(self, screen_size):
        self.scsz = screen_size
        self.aspect_ratio = self.scsz[0]/self.scsz[1]  # aspect ratio of screen
        self.real_ratio = self.scsz[1]/self.height  # ratio of actual height to screen height

    def point_to_float(self, point_num):
        # converts a point to a float 0 - 1
        x = self.points[point_num].pos[0] / (self.aspect_ratio * self.height)
        y = self.points[point_num].pos[1] / self.height
        return x, y

    def avg_points_to_float(self, point_nums):
        # converts a list of point indices to their average position as a float 0 - 1
        x = 0
        y = 0
        for num in point_nums:
            coords = self.point_to_float(num)
            x += coords[0]
            y += coords[1]
        x /= len(point_nums)
        y /= len(point_nums)
        return x, y

    def points_to_real(self, point_nums):
        # converts a list of point indices to their corresponding real coords as a flattened list
        data = []
        for num in point_nums:
            data += [self.points[num].pos[0] * self.real_ratio,
                     self.points[num].pos[1] * self.real_ratio]
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
        for f in range(len(self.points)):
            self.p_data += self.points_to_real((f,))
            if self.point_color.color_type == 2:
                self.pc_data += self.point_color.get_color(self.point_to_float(f))
            else:
                self.pc_data += self.point_color.get_color(f)
            for i in self.lines[f]:
                self.l_data += self.points_to_real((f, i))
                if self.line_color.color_type == 2:
                    if self.color_fade:
                        self.lc_data += self.line_color.get_color(self.point_to_float(f))
                        self.lc_data += self.line_color.get_color(self.point_to_float(i))
                    else:
                        self.lc_data += 2 * self.line_color.get_color(self.avg_points_to_float((f, i)))
                else:
                    if self.color_fade:
                        self.lc_data += self.line_color.get_color(tuple(sorted((f, i))), number=2)
                    else:
                        self.lc_data += 2 * self.line_color.get_color(tuple(sorted((f, i))))
        for tri in self.triangles:
            self.tr_data += self.points_to_real(tri)
            if self.triangle_color.color_type == 2:
                if self.color_fade:
                    for t in tri:
                        self.trc_data += self.triangle_color.get_color(self.point_to_float(t))
                else:
                    self.trc_data += 3 * self.triangle_color.get_color(self.avg_points_to_float(tri))
            else:
                if self.color_fade:
                    self.trc_data += self.triangle_color.get_color(tri, number=3)
                else:
                    self.trc_data += 3 * self.triangle_color.get_color(tri)

    def draw(self, points=True, lines=False, triangles=False):
        if triangles:
            pyglet.graphics.draw(len(self.tr_data)//2, pyglet.gl.GL_TRIANGLES,
                                 ('v2f', self.tr_data), ('c3B', self.trc_data))
        if lines:
            pyglet.graphics.draw(len(self.l_data)//2, pyglet.gl.GL_LINES,
                                 ('v2f', self.l_data), ('c3B', self.lc_data))
        if points:
            pyglet.graphics.draw(len(self.points), pyglet.gl.GL_POINTS,
                                 ('v2f', self.p_data), ('c3B', self.pc_data))


class GUI(pyglet.window.Window):

    def __init__(self):
        title = 'point interactions'
        super(GUI, self).__init__(caption=title, fullscreen=True, resizable=True)
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.set_minimum_size(100, 100)
        pyglet.gl.glReadBuffer(pyglet.gl.GL_FRONT)
        pyglet.gl.glDrawBuffer(pyglet.gl.GL_BACK)

        # set parameters
        if params.behavior_preset == "mesh":
            self.dots = Field(100, 100, 2, 0.9, 1, (lambda x: 1/3/x,), self.get_size())
        elif params.behavior_preset == "blob":
            self.dots = Field(100, 100, 4, 0.9, 1, (lambda x: 2-x/5,), self.get_size())
        elif params.behavior_preset == "shrink":
            self.dots = Field(100, 10, 5, 0.9, 1, (lambda x: -math.atan(x)/10,), self.get_size())
        elif params.behavior_preset == "fireworks":
            self.dots = Field(100, 100, 2, 0.9, 1, (lambda x: -1/x/50,), self.get_size())
        elif params.behavior_preset == "amoeba":
            self.dots = Field(100, 100, 4, 0.9, 1, (lambda x: 1.2-x/5, lambda x: 2.4-x/5),
                              self.get_size(), symmetric=False)
        else:
            if type(params.point_function) != tuple:
                params.point_function = (params.point_function,)
            self.dots = Field(params.number_of_points, 100, params.point_connections,
                              params.air_resistance, params.bounciness, params.point_function,
                              self.get_size(), symmetric=params.symmetric_forces)

        pyglet.gl.glClearColor(params.background_color[0]/255, params.background_color[1]/255,
                               params.background_color[2]/255, 1.0)
        pyglet.gl.glPointSize(params.point_size)
        pyglet.gl.glLineWidth(params.line_width)

        mouse_map = {"left": pyglet.window.mouse.LEFT, "right": pyglet.window.mouse.RIGHT,
                     "middle": pyglet.window.mouse.MIDDLE}
        self.active_button = mouse_map[params.mouse_button.lower()]
        self.full = True
        self.pause = False
        self.stain = False
        self.fps_show = False
        self.point_show = True
        self.line_show = False
        self.triangle_show = False
        self.slow_down = 1

    def on_key_press(self, symbol, modifiers):
        key_str = key.symbol_string(symbol)
        if key_str == params.fullscreen_key:  # toggle fullscreen
            self.full = not self.full
            self.set_fullscreen(self.full)
        elif key_str == params.pause_key:  # pause
            self.pause = not self.pause
        elif key_str == params.force_frame_key:  # go forward one frame
            self.dots.update(num=self.slow_down)
        elif key_str == params.stain_key:  # toggle stain
            self.stain = not self.stain
        elif key_str == params.fps_key:  # toggle fps reading
            self.fps_show = not self.fps_show
        elif key_str == params.reset_key:  # reset field
            self.dots.reset()
        elif key_str == params.slow_key:  # make the animation go slow
            self.slow_down = params.slow_down - self.slow_down
        elif key_str == params.point_key:  # show points
            self.point_show = not self.point_show
        elif key_str == params.line_key:  # show lines
            self.line_show = not self.line_show
        elif key_str == params.triangle_key:  # show any triangles that hae formed
            self.triangle_show = not self.triangle_show
        elif key_str == params.reseed_key:  # re-seed the random colors
            self.dots.reseed()
        elif key_str == params.color_fade_key:  # re-seed the random colors
            self.dots.change_fade()
        elif key_str == params.capture_key:  # capture the image on screen into a file
            pyglet.image.get_buffer_manager().get_color_buffer().save(get_new_file_name(params.path_name))
        elif key_str == params.quit_key:  # exit
            self.close()

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

    def update(self, dt):
        if not self.pause:
            self.dots.update(num=self.slow_down)
        if not self.stain:
            pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
        else:
            # copy front buffer to back buffer for reliable staining
            pyglet.gl.glCopyPixels(0, 0, self.width, self.height, pyglet.gl.GL_COLOR)
        self.dots.draw(points=self.point_show, lines=self.line_show, triangles=self.triangle_show)
        if self.fps_show:
            self.fps_display.draw()


if __name__ == "__main__":
    # just make double sure that there are no path issues
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(params.path_name):
        os.mkdir(params.path_name)

    # run the program
    window = GUI()
    pyglet.clock.schedule_interval(window.update, 1/params.FPS)
    pyglet.app.run()
