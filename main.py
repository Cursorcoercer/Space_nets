import pyglet
from pyglet.window import key
import random
import math


def dist_sq(coords_1, coords_2):
    # find the distance between two coordinates
    return (coords_1[0] - coords_2[0])**2 + (coords_1[1] - coords_2[1])**2


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


class Field:

    def __init__(self, point_num, height, feel_num, air_resistance, bounciness, func, screen_size, init_vel=0):
        self.num = point_num
        self.height = height
        self.feel_num = feel_num
        self.air = air_resistance
        self.bounce = bounciness
        self.func = func
        self.scsz = screen_size
        self.init_vel = init_vel
        self.aspect_ratio = self.scsz[0]/self.scsz[1]  # aspect ratio of screen
        self.real_ratio = self.scsz[1]/self.height  # ratio of actual height to screen height
        self.update_num = 0
        self.points = []
        self.lines = []
        self.p_data = []
        self.pc_data = []
        self.l_data = []
        self.lc1_data = []
        self.lc_data = []
        self.grabbed = None
        self.last_pos = (0, 0)
        self.point_color = [255, 255, 255]
        self.default_line_color = [255, 255, 255]
        self.special_line_color_1 = [255, 110, 110]
        self.special_line_color_2 = [70, 255, 120]
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
        self.pc_data = self.num * self.point_color
        self.lc1_data = (2 * self.num * self.feel_num) * self.default_line_color
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
        if velocity is not None:
            self.points[self.grabbed].vel[0] = velocity[0] / self.real_ratio
            self.points[self.grabbed].vel[1] = velocity[1] / self.real_ratio

    def out_of_bounds(self, position):
        # return a boolean tuple with out of bounds for horizontal and vertical
        return (position[0] < 0 or self.height * self.aspect_ratio < position[0],
                position[1] < 0 or self.height < position[1])

    def update_vel(self):
        # forces from other points
        for f in range(len(self.points)):
            closest = self.find_closest(f)
            self.lines[f] = []
            for c in closest:
                self.lines[f].append(c[0])
                act_dist = math.sqrt(c[1])
                force = self.func(act_dist)
                x_force = force * (self.points[f].pos[0] - self.points[c[0]].pos[0]) / act_dist
                y_force = force * (self.points[f].pos[1] - self.points[c[0]].pos[1]) / act_dist
                self.points[f].vel[0] += x_force
                self.points[f].vel[1] += y_force
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

    def update(self, num=1):
        # first we update the velocities of the points based on their forces
        self.update_num = (self.update_num + 1) % num
        if self.update_num == 1 or num == 1:
            self.update_vel()
        for f in range(len(self.points)):
            # then we update the position of the points based on their velocities
            self.points[f].move(div=num)
            # then we account for any points that somehow slipped out of bounds
            if any(self.out_of_bounds(self.points[f].pos)):
                self.points[f] = self.new_point()
        # then we update the p_data for the vertex buffer
        self.move_grabbed()
        self.prepare_data()

    def prepare_data(self):
        # transforms p_data into screen coordinates
        # then puts it in proper openGL type
        self.p_data = []
        self.l_data = []
        self.lc_data = []
        for f in range(len(self.points)):
            self.p_data += [self.points[f].pos[0] * self.real_ratio,
                            self.points[f].pos[1] * self.real_ratio]
            for i in self.lines[f]:
                self.l_data += [self.points[f].pos[0] * self.real_ratio,
                                self.points[f].pos[1] * self.real_ratio,
                                self.points[i].pos[0] * self.real_ratio,
                                self.points[i].pos[1] * self.real_ratio]
                if f in self.lines[i]:
                    self.lc_data += 2*self.special_line_color_2
                else:
                    self.lc_data += 2*self.special_line_color_1

    def draw(self, lines=False, colored=False):
        pyglet.graphics.draw(len(self.points), pyglet.gl.GL_POINTS,
                             ('v2f', self.p_data), ('c3B', self.pc_data))
        if lines:
            if colored:
                pyglet.graphics.draw(len(self.l_data)//2, pyglet.gl.GL_LINES,
                                     ('v2f', self.l_data), ('c3B', self.lc_data))
            else:
                pyglet.graphics.draw(len(self.l_data)//2, pyglet.gl.GL_LINES,
                                     ('v2f', self.l_data), ('c3B', self.lc1_data))


class GUI(pyglet.window.Window):

    def __init__(self):
        title = 'point interactions'
        config = pyglet.gl.Config(double_buffer=False)
        super(GUI, self).__init__(caption=title, fullscreen=True, config=config, vsync=0)
        self.fps_display = pyglet.window.FPSDisplay(window=self)

        # self.dots = Field(100, 100, 2, 0.9, 1, lambda x: 1/(4*x), self.get_size(), init_vel=0)
        self.dots = Field(100, 100, 4, 0.9, 1, lambda x: (10 - x) / 5, self.get_size(), init_vel=0)

        self.background_color = (0, 0, 0)  # set the background color hex
        self.point_size = 2  # set the point size in pixels

        self.pause = False
        self.stain = False
        self.fps_show = False
        self.line_show = False
        self.line_colors = False
        self.slow_down = 1
        
        self.on_start()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            # press space to pause
            self.pause = not self.pause
        elif symbol == key.N:
            # press N to go forward one frame
            self.dots.update()
        elif symbol == key.S:
            # press S to toggle stain
            self.stain = not self.stain
        elif symbol == key.F:
            # press F to toggle fps reading
            self.fps_show = not self.fps_show
        elif symbol == key.R:
            # press R to reset field
            self.dots.reset()
        elif symbol == key.L:
            # press L to show lines
            self.line_show = not self.line_show
        elif symbol == key.C:
            # press C to color code the lines
            self.line_colors = not self.line_colors
        elif symbol == key.W:
            # press W to make the animation go slow
            self.slow_down = 10 - self.slow_down
        elif symbol == key.ESCAPE:
            # press escape to exit
            pyglet.app.exit()

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.RIGHT:
            self.dots.set_grabbed((x, y))
            self.dots.move_grabbed((x, y))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.dots.move_grabbed((x, y), velocity=(dx, dy))

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.RIGHT:
            self.dots.set_grabbed(None)

    def on_start(self):
        # do stuff to get the screen ready
        pyglet.gl.glClearColor(self.background_color[0]/255, self.background_color[1]/255,
                               self.background_color[2]/255, 1.0)
        pyglet.gl.glPointSize(self.point_size)

    def update(self, dt):
        if not self.pause:
            self.dots.update(num=self.slow_down)
        if not self.stain:
            pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
        self.dots.draw(lines=self.line_show, colored=self.line_colors)
        if self.fps_show:
            self.fps_display.draw()


if __name__ == "__main__":
    window = GUI()
    FPS = 60
    pyglet.clock.schedule_interval(window.update, 1/FPS)
    pyglet.app.run()
