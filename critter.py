import grid
import nengo
import nengo.spa as spa
import numpy as np 
import itertools

mymap="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""
highly_complex_map = """
#######
# M   #
# # # #
# #B# #
# # # #
# Y # #
#  R  #
#   # #
# G G #
#######
"""
complex_map = """
#######
# M   #
# # # #
# #B# #
#  #  #
# Y # #
#  R  #
#     #
#  G  #
#######
"""
large_map = """
#######################
# M  # #  #  #  #     #
# #  #  # # #  # # #  #
# #  #  #  #B#   #    #
# #   # #  #  #  # #  #
#G      #  #    # #   #
#  #  # Y #        #  #
#  #  #    #  #  R    #
#  # #  #  #    #     #
#       B#    #  #  G #
#######################
"""
detailed_map = """
########################
# M   #   #   #   #    #
# #B# # # #   # # # #  #
#   #  #  #  # #  # #  #
#      #  #   #     #  #
#  #  #   # #   # Y #  #
#  #  #  #  #  #    #  #
#  #    # # #      #   #
#  # #B#  #  #     R   #
#  # #  # #  #     #   #
#  #   #  #   #    #   #
#  # #   #    #  #  G  #
########################
"""


class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
             
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
            
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5
            
            

def move(t, x):
    speed, rotation = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    body.go_forward(speed * dt * max_speed)

def detect(t):
    angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
    return [body.detect(d, max_distance=4)[0] for d in angles]


def cell2rgb(t):
    
    c = col_values.get(body.cell.cellcolor)
    noise = np.random.normal(0, noise_val,3)
    c = np.clip(c + noise, 0, 1)
    
    return c

def movement_func(x):
    turn = x[2] - x[0]
    spd = x[1] - 0.7
    return spd, turn

def look_ahead(t):
    
    done = False
    
    cell = body.cell.neighbour[int(body.dir)]
    if cell.cellcolor > 0:
        done = True 
        
    while cell.neighbour[int(body.dir)].wall == False and not done:
        cell = cell.neighbour[int(body.dir)]
        
        if cell.cellcolor > 0:
            done = True
    
    c = col_values.get(cell.cellcolor)
    noise = np.random.normal(0, noise_val,3)
    c = np.clip(c + noise, 0, 1)
    
    return c
    
# -------------- My Functions ----------------------------
def mov_expl(x):
    rand = np.random.uniform(-1,1)
    spd = max_speed*(max(0,1-x[0]))
    rot = rand*(1-x[0])
    return [spd, rot]
    
def mov_scan(x):
    spd = 0.1 
    rot = (1/x[0])*0.5
    return [spd, rot]

def learning_phases(t):
    x = np.zeros(2)
    if t > 10000:
        x[0] = -1
        x[1] = 1
    return x
    
def gate_error(t, x):
    phase_enable = x[0]
    error = x[1:]
    return error*phase_enable

col_values = {
    0: [0.9, 0.9, 0.9], # White
    1: [0.2, 0.8, 0.2], # Green
    2: [0.8, 0.2, 0.2], # Red
    3: [0.2, 0.2, 0.8], # Blue
    4: [0.8, 0.2, 0.8], # Magenta
    5: [0.8, 0.8, 0.2], # Yellow
}
# hyperparameters 
lr = 1e-3


# model parameters

noise_val = 0.1 # how much noise there will be in the colour info
# D_state = 32
# D_place = 6
# D_mem = 2*D_state + D_place
max_speed = 3



# # initiate color vocab for SPA
# col_string = [["WHITE", [0.9,0.9,0.9]],
#             ["GREEN",[0.2,0.8,0.2]],
#             "RED","BLUE","MAGENTA","YELLOW"]
# col_combinations = itertools.product(col_string, repeat = 2)


# col_vocab = spa.Vocabulary(D_state)

voja = nengo.Voja(lr)
pes = nengo.PES(lr)

# initiate model components
world = grid.World(Cell, map=large_map, directions=int(4))

body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)


model = nengo.Network()
with model:
    
    env = grid.GridNode(world, dt=0.005)
    # -------------- Module ---------------
    # sensors
    proximity_sensors = nengo.Node(detect) # returns dist [-0.5,0,0.5] relative to facing direction
    current_color = nengo.Node(cell2rgb) # returns rgb values of current cell
    ahead_color = nengo.Node(look_ahead) # returns rgb values of closest colored cell
        
    # perceptual modules
    color_cur = nengo.Ensemble(n_neurons=50, dimensions=3)
    color_ah = nengo.Ensemble(n_neurons=50, dimensions=3)
    walldist = nengo.Ensemble(n_neurons=50, dimensions=3)

    
    # derivatives
    ders = nengo.Ensemble(n_neurons=50, dimensions=3)
    
    # grid cells with custom encoders - randomly sampled btw 0 and 1, scaled to multiple levels
    grid_cells = nengo.Ensemble(n_neurons=1000, 
                                dimensions = 6, 
                                encoders=nengo.dists.Uniform(0, 1).sample(1000, 6) * np.random.choice([0.5, 1.0, 2.0], size=(1000, 1)))
    
    # place cells for uniquely representing location
    place_cells = nengo.Ensemble(n_neurons=500, dimensions = 12)

    # color sequence memory
    col_seq_int = nengo.Ensemble(n_neurons = 1000, dimensions = 6)
    col_place_enc = nengo.Ensemble(n_neurons=1000, dimensions = 18)
    col_pred = nengo.Ensemble(n_neurons=500, dimensions = 3)
    error_col = nengo.Ensemble(n_neurons=50, dimensions=3)
    
    
    # basic action selection
    cb = nengo.Ensemble(n_neurons = 50, dimensions=2)

    # movement modules
    expl = nengo.Ensemble(n_neurons = 50, dimensions = 1)
    scan = nengo.Ensemble(n_neurons = 50, dimensions = 1)

    mov_out = nengo.Ensemble(n_neurons = 50, dimensions = 2)

    # Action Output
    movement = nengo.Node(move, size_in=2)
    
    # learning phases
    gate_node = nengo.Node(gate_error, size_in = 4, size_out=3)
    phase_node = nengo.Node(learning_phases, size_out = 2)
    
    
    # -------------------- Connections ----------------
    # Connections from sensors to sensory ensembles
    nengo.Connection(proximity_sensors, walldist)
    nengo.Connection(current_color, color_cur)
    nengo.Connection(ahead_color, color_ah)
    
    # recurrent connections for sensory stabilisation
    nengo.Connection(color_cur, color_cur, synapse=0.02)
    nengo.Connection(color_ah, color_ah, synapse=0.02)
    nengo.Connection(walldist, walldist, synapse=0.02)
    
    # double for computing derivative
    nengo.Connection(walldist, ders)
    nengo.Connection(walldist, ders, transform = -1, synapse=0.01)
    
    # walldist and derivative to grid int
    nengo.Connection(walldist, grid_cells[0:3])
    nengo.Connection(ders, grid_cells[3:6])
    

    
    # grid to place
    conn_1 = nengo.Connection(grid_cells, place_cells[0:6])
    conn_2 = nengo.Connection(color_cur, place_cells[6:9])
    conn_3 = nengo.Connection(color_ah, place_cells[9:12])

    
    # memory connections
    nengo.Connection(color_cur, col_seq_int[0:3])
    nengo.Connection(color_ah, col_seq_int[0:3])
    col_seq_rep = nengo.Connection(col_seq_int, col_place_enc[0:6], learning_rule_type=voja)
    place_enc = nengo.Connection(place_cells, col_place_enc[6:18], learning_rule_type=voja)
    col_pred_enc = nengo.Connection(col_place_enc, col_pred, transform = np.random.randn(3,18), learning_rule_type = pes) 
    
    
    # prediction connections
    nengo.Connection(col_pred, error_col)
    nengo.Connection(color_ah, error_col, transform = -1, synapse=0.05)
    
    
    # #action selection
    nengo.Connection(error_col, cb[0], function = lambda x: np.linalg.norm(np.array(x)))
    nengo.Connection(cb[0], expl[0], transform=-1)
    nengo.Connection(cb[0], scan[0])
    
    # movement connections
    nengo.Connection(expl, mov_out, function=mov_expl)
    nengo.Connection(scan, mov_out, function=mov_scan)
    nengo.Connection(walldist, mov_out, function=movement_func)
    nengo.Connection(mov_out, movement)
    
    # movement recurrency to stabilise
    nengo.Connection(mov_out, mov_out, synapse=0.005)
    
    
    # learning phases connections 
    nengo.Connection(phase_node[0], place_enc.learning_rule)
    nengo.Connection(phase_node[0], col_seq_rep.learning_rule)
    nengo.Connection(phase_node[1], gate_node[0])
    nengo.Connection(error_col, gate_node[1:])
    
    nengo.Connection(gate_node, col_pred_enc.learning_rule) 
    



    
    
    