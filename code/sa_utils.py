import numpy as np
import matplotlib.pyplot as plt

def plot_f(f, min_x, max_x):
    x_axis = np.linspace(min_x, max_x, 100)
    fig, ax = plt.subplots()

    y = [f(x) for x in x_axis]

    ax.plot(x_axis, y, linewidth=3, label='f', c='b')

    # set the spine locations
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # turn off the top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.grid(True, which='both')

    ax.legend()

    return ax

class Node:

    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent

    def __repr__(self):
        return "Node: {}".format(self.state)

    def path(self):
        current = self
        path_back = [current]
        while current.parent is not None:
            path_back.append(current.parent)
            current = current.parent
        return reversed(path_back)

    def expand(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

class OneDNode(Node):

    _value_f = lambda _: None
    _actions = []
    _boundaries = []

    def __init__(self, state, parent = None):
        super().__init__(state, parent)

    def expand(self):
        children = []
        for action in OneDNode._actions:
            new_state_value = self.state + action
            if len(OneDNode._boundaries) == 0 or (new_state_value >= OneDNode._boundaries[0] and new_state_value <= OneDNode._boundaries[1]):
                child = OneDNode(new_state_value, parent = self)
                children.append(child)

        return children

    def value(self):
        return OneDNode._value_f(self.state)

class TwoDNode(Node):

    _value_f = lambda _: None

    def __init__(self, state, parent = None):
        super().__init__(state, parent)

    def expand(self):
        children = []

        # four children: N, S, E, W
        children.append(TwoDNode(state = [self.state[0]+1, self.state[1]], parent = self))
        children.append(TwoDNode(state = [self.state[0]-1, self.state[1]], parent = self))
        children.append(TwoDNode(state = [self.state[0], self.state[1]+1], parent = self))
        children.append(TwoDNode(state = [self.state[0], self.state[1]-1], parent = self))

        return children

    def value(self):
        return TwoDNode._value_f(self.state)

def simulated_annealing(initial_n, temp_schedule, max_iter, random_state = np.random.RandomState(0)):
    current_n = initial_n
    for t in range(max_iter):

        T = temp_schedule(t)
        next_nodes = current_n.expand()

        if len(next_nodes) == 0:
            return current_n
        else:
            next_n = random_state.choice(next_nodes)

            delta_e =  next_n.value() - current_n.value()

            if delta_e > 0:
                current_n = next_n
            else:
                p = np.exp(delta_e/T)
                #print("{:.1f} -> {:.1f}: {:.3f}".format(current_n.state, next_n.state, p))
                if random_state.random() < p:
                    current_n = next_n
    return current_n

def hill_climbing(initial_n):
    current_n = initial_n
    current_best = initial_n.value()

    while True:
        next_nodes = current_n.expand()
        next_vals = [node.value() for node in next_nodes]
        max_index = np.argmax(next_vals)
        if next_vals[max_index] > current_best:
            current_best = next_vals[max_index]
            current_n = next_nodes[max_index]
        else:
            return current_n
