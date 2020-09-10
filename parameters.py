import math

# this file exists to make it more user friendly to tweak the parameters of this program
# be sure to save this file in the same directory as the main file, with the name "parameters"

# ----- dot behavior settings -----
# first you can choose between some preset behaviors
# if you'd like to set your own, set this value to None
# if this is not None it invalidates everything between here and the control settings
# possible presets: "mesh", "blob", "shrink", "fireworks", "amoeba"
behavior_preset = None

number_of_points = 100  # the number of points
point_connections = 4  # the number of connections each point will make
air_resistance = 0.9  # the amount of air resistance, 1 is no resistance, 0 is no movement
bounciness = 1  # how bouncy the walls are, 1 conserves velocity, 0 is no bounce
symmetric_forces = True  # determines whether or not point forces are symmetric True or False


# point interaction function(s)
# this determines how the dots interact, if multiple functions are passed, they will be assigned to points at random
# here are various functions that I find make interesting interactions
def repel(divider):
    return lambda x: 1/(divider * x)


def repel_t(divider):
    # non-asymptotic repelling
    return lambda x: math.atan(x)/divider


def hold(distance, divider):
    # the dots hold each other at a particular distance
    return lambda x: (distance - x)/divider


# set the function(s)
# some examples: hold(10, 5) or hold(6, 5), hold(12, 5) or repel(4)
point_function = hold(10, 5)


# ----- control settings -----
FPS = 60  # the maximum frame rate to run at
fullscreen_key = 'ENTER'  # the key that toggles fullscreen mode
pause_key = 'SPACE'  # the key that pauses the simulation
force_frame_key = 'N'  # the key that moves the simulation exactly one frame while paused
stain_key = 'S'  # the key that toggles whether or not the screen gets cleared
fps_key = 'F'  # the key that toggles the on-screen fps
reset_key = 'R'  # the key that resets all points to random positions
slow_key = 'W'  # the key that toggles slow mode
slow_down = 10  # how many times slower slow mode is than normal
point_key = 'P'  # the key that toggles point visibility
line_key = 'L'  # the key that toggles line visibility
triangle_key = 'T'  # the key that toggles triangle visibility
reseed_key = 'B'  # the key that re-randomizes any randomly chosen colors
color_fade_key = 'C'  # the key that switches the color mode from static to fade
capture_key = 'K'  # the key that captures what's on screen into an image
path_name = 'screenshots'  # the name of the directory to save screenshots to
quit_key = 'ESCAPE'  # the key that closes the window
mouse_button = 'RIGHT'  # the mouse button that allows you to manipulate the dots


# ----- color settings -----
# use these to alter colors/color palettes
# example color (255, 0, 0)
# example palette ((255, 0, 0), (0, 255, 0), (0, 0, 255))
# example color map r"path\folder\file_name.png"
# example k means r"path\folder\file_name.png", 5
# or set to None for random colors
background_color = (0, 0, 0)  # the color of the background, must be one color
point_color = (255, 255, 255)  # the color of the points
line_color = None  # the color of the lines
triangle_color = ((255, 176, 176), (254, 113, 113), (51, 93, 45), (126, 160, 77))  # the color of the triangles


# ----- other aesthetics -----
point_size = 2  # set the size of the points in pixels
line_width = 1  # set the width of the lines in pixels

# these are parameters that help k-means run faster, probably don't worry about them
pre_means_sorting_smoothness = 8  # a constant to smooth out the step sort
pre_means_clustering = 100  # a higher number yields more accurate k-means results at the expense of time

