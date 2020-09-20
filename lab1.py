# MIT 6.034 Lab 1: Search
# Written by 6.034 staff

from search import Edge, UndirectedGraph, do_nothing_fn, make_generic_search
import read_graphs
from functools import reduce
from collections import deque

all_graphs = read_graphs.get_graphs()
GRAPH_0 = all_graphs['GRAPH_0']
GRAPH_1 = all_graphs['GRAPH_1']
GRAPH_2 = all_graphs['GRAPH_2']
GRAPH_3 = all_graphs['GRAPH_3']
GRAPH_FOR_HEURISTICS = all_graphs['GRAPH_FOR_HEURISTICS']


# Please see wiki lab page for full description of functions and API.

#### PART 1: Helper Functions ##################################################

def path_length(graph, path):
    """Returns the total length (sum of edge weights) of a path defined by a
    list of nodes coercing an edge-linked traversal through a graph.
    (That is, the list of nodes defines a path through the graph.)
    A path with fewer than 2 nodes should have length of 0.
    You can assume that all edges along the path have a valid numeric weight."""
    
    # Check if path less than 2 nodes
    if len(path) < 2:
        return 0
    
    # Tracks distance, initialized at 0
    distance = 0

    # Iterates through adjacent node pairs 
    # Gets the edge object using graph.get_edge 
    # Use edge.length to acquire distances
    # Assumes all nodes are connected
    for i in range(len(path)-1):
        distance += (graph.get_edge(path[i], path[i+1])).length

    return distance

def has_loops(path):
    """Returns True if this path has a loop in it, i.e. if it
    visits a node more than once. Returns False otherwise."""
    
    # Use sets and compare length. If there's size discrepency, then a node repeated
    unique_nodes = set(path)

    if len(unique_nodes) < len(path):
        return True

    return False

def extensions(graph, path):
    """Returns a list of paths. Each path in the list should be a one-node
    extension of the input path, where an extension is defined as a path formed
    by adding a neighbor node (of the final node in the path) to the path.
    Returned paths should not have loops, i.e. should not visit the same node
    twice. The returned paths should be sorted in lexicographic order."""
    
    # Since we're look at extensions, we need to consider the last element of the path
    # We can use graph.get_neighbors(node) to find neigbors
    # Want to exclude any visited nodes (can use sets to keep track)

    # All the visited nodes
    visited_nodes = set(path)

    # Current node on path
    curr_node = path[-1]

    # Getting neighbors
    neighbors = graph.get_neighbors(curr_node)

    # Make extension array and add to it from neighbors
    extensions = []
    for neighbor in neighbors:
        if neighbor not in visited_nodes:
            extensions.append(neighbor)

    # Sort the nodes
    extensions.sort()

    # Construct the paths 
    paths = []
    for extension in extensions:
        path_copy = path.copy() # avoid tampering with original path
        path_copy.append(extension)
        paths.append(path_copy)
    
    return paths

def sort_by_heuristic(graph, goalNode, nodes):
    """Given a list of nodes, sorts them best-to-worst based on the heuristic
    from each node to the goal node. Here, and in general for this lab, we
    consider a smaller heuristic value to be "better" because it represents a
    shorter potential path to the goal. Break ties lexicographically by 
    node name."""
    
    # Iterate through nodes, keep track of heuristic in dictionary and sort
    heuristic_values = {}
    for node in nodes:
        heuristic_val = graph.get_heuristic_value(node, goalNode)
        if heuristic_val in heuristic_values.keys():
            heuristic_values[heuristic_val].append(node)
        else:
            heuristic_values[heuristic_val] = [node]

    # Sorts the dictionary based on hueristic values
    keys = sorted(heuristic_values.keys())

    # Apply the lexical sort and append
    sorted_nodes = []
    for key in keys:
        nodes_lst = sorted(heuristic_values[key])
        sorted_nodes.extend(nodes_lst)

    return sorted_nodes


# You can ignore the following line.  It allows generic_search (PART 3) to
# access the extensions and has_loops functions that you just defined in PART 1.
generic_search = make_generic_search(extensions, has_loops)  # DO NOT CHANGE


#### PART 2: Basic Search ######################################################

def basic_dfs(graph, startNode, goalNode):
    """
    Performs a depth-first search on a graph from a specified start
    node to a specified goal node, returning a path-to-goal if it
    exists, otherwise returning None.
    Uses backtracking, but does not use an extended set.
    """

    queue = [[startNode]]

    # Loops till queue exhausted
    while queue:
        # Check if current path suffics
        if queue[0][-1] == goalNode:
            return queue[0]

        # get extensions of current path
        possible_paths = extensions(graph, queue[0])

        # pop and replace the first elements of the queue
        queue.pop(0)
        queue = possible_paths + queue 

    return None

def basic_bfs(graph, startNode, goalNode):
    """
    Performs a breadth-first search on a graph from a specified start
    node to a specified goal node, returning a path-to-goal if it
    exists, otherwise returning None.
    """
    
    queue = [[startNode]]

    # Loops till queue exhausted
    while queue:
        # Check if current path suffics
        if queue[0][-1] == goalNode:
            return queue[0]

        # get extensions of current path
        possible_paths = extensions(graph, queue[0])

        # pop and append new paths to the end of the queue
        queue.pop(0)
        queue = queue + possible_paths 

    return None


#### PART 3: Generic Search ####################################################

# Generic search requires four arguments (see wiki for more details):
# sort_new_paths_fn: a function that sorts new paths that are added to the agenda
# add_paths_to_front_of_agenda: True if new paths should be added to the front of the agenda
# sort_agenda_fn: function to sort the agenda after adding all new paths 
# use_extended_set: True if the algorithm should utilize an extended set


# Define your custom path-sorting functions here.
# Each path-sorting function should be in this form:
# def my_sorting_fn(graph, goalNode, paths):
#     # YOUR CODE HERE
#     return sorted_paths

# Lexicographical sorting (DFS and BFS)
def lexical_sorting_fn(graph, goalNode, paths):
    return sorted(paths)

# Hueristic: sort by highest heuristic (Hill Climbing and Best First)
def heuristic_sorting_fn(graph, goalNode, paths):
    paths_dict = {}
    for path in paths:
        curr_node = path[-1]
        if curr_node in paths_dict.keys():
            paths_dict[curr_node].append(path)
        else:
            paths_dict[curr_node] = [path]

    curr_nodes = paths_dict.keys()
    sorted_curr_nodes = sort_by_heuristic(graph, goalNode, curr_nodes)

    sorted_paths = []
    for node in sorted_curr_nodes:
        path_lst = sorted(paths_dict[node])
        sorted_paths.extend(path_lst)

    return sorted_paths

# Path Length: sort by length of path (Branch and Bound)
def length_sorting_fn(graph, goalNode, paths):
    paths_dict = {}
    for path in paths:
        length = path_length(graph, path)
        if length in paths_dict.keys():
            paths_dict[length].append(path)
        else:
            paths_dict[length] = [path]

    lengths = sorted(paths_dict.keys())

    sorted_paths = []
    for length in lengths:
        path_lst = sorted(paths_dict[length])
        sorted_paths.extend(path_lst)

    return sorted_paths

# Path Length and Heuristic: sort by length of path and hueristics (Branch and Bound with Heuristics)
def length_heuristic_sorting_fn(graph, goalNode, paths):
    paths_dict = {}
    for path in paths:
        length = path_length(graph, path)
        heuristic = graph.get_heuristic_value(path[-1], goalNode)
        metric = length + heuristic
        if metric in paths_dict.keys():
            paths_dict[metric].append(path)
        else:
            paths_dict[metric] = [path]

    metrics = sorted(paths_dict.keys())

    sorted_paths = []
    for metric in metrics:
        path_lst = sorted(paths_dict[metric])
        sorted_paths.extend(path_lst)

    return sorted_paths


generic_dfs = [lexical_sorting_fn, True, do_nothing_fn, False]

generic_bfs = [lexical_sorting_fn, False, do_nothing_fn, False]

generic_hill_climbing = [heuristic_sorting_fn, True, do_nothing_fn, False]

generic_best_first = [do_nothing_fn, True, heuristic_sorting_fn, False]

generic_branch_and_bound = [do_nothing_fn, True, length_sorting_fn, False]

generic_branch_and_bound_with_heuristic = [do_nothing_fn, True, length_heuristic_sorting_fn, False]

generic_branch_and_bound_with_extended_set = [do_nothing_fn, True, length_sorting_fn, True]

generic_a_star = [do_nothing_fn, True, length_heuristic_sorting_fn, True]


# Here is an example of how to call generic_search (uncomment to run):
# my_dfs_fn = generic_search(*generic_dfs)
# my_dfs_path = my_dfs_fn(GRAPH_2, 'S', 'G')
# print(my_dfs_path)

# Or, combining the first two steps:
# my_dfs_path = generic_search(*generic_dfs)(GRAPH_2, 'S', 'G')
# print(my_dfs_path)


### OPTIONAL: Generic Beam Search

# If you want to run local tests for generic_beam, change TEST_GENERIC_BEAM to True:
TEST_GENERIC_BEAM = False

# The sort_agenda_fn for beam search takes fourth argument, beam_width:
# def my_beam_sorting_fn(graph, goalNode, paths, beam_width):
#     # YOUR CODE HERE
#     return sorted_beam_agenda

generic_beam = [None, None, None, None]


# Uncomment this to test your generic_beam search:
# print(generic_search(*generic_beam)(GRAPH_2, 'S', 'G', beam_width=2))


#### PART 4: Heuristics ########################################################

def is_admissible(graph, goalNode):
    """Returns True if this graph's heuristic is admissible; else False.
    A heuristic is admissible if it is either always exactly correct or overly
    optimistic; it never over-estimates the cost to the goal."""
    
    # Get all the nodes
    nodes = graph.nodes

    # Branch and Bound with Extended Set (BBES) is the fastest and reassures shortest path
    bbes = generic_search(*generic_branch_and_bound_with_extended_set)

    # Iterate through nodes, get distance and heuristic, and compare
    for node in nodes:
        shortest_path = bbes(graph, node, goalNode)
        actual_dist = path_length(graph, shortest_path)
        heuristic_dist = graph.get_heuristic_value(node, goalNode)
        if heuristic_dist > actual_dist:
            return False

    return True

def is_consistent(graph, goalNode):
    """Returns True if this graph's heuristic is consistent; else False.
    A consistent heuristic satisfies the following property for all
    nodes v in the graph:
        Suppose v is a node in the graph, and N is a neighbor of v,
        then, heuristic(v) <= heuristic(N) + edge_weight(v, N)
    In other words, moving from one node to a neighboring node never unfairly
    decreases the heuristic.
    This is equivalent to the heuristic satisfying the triangle inequality."""
    raise NotImplementedError


### OPTIONAL: Picking Heuristics

# If you want to run local tests on your heuristics, change TEST_HEURISTICS to True.
#  Note that you MUST have completed generic a_star in order to do this:
TEST_HEURISTICS = False


# heuristic_1: admissible and consistent

[h1_S, h1_A, h1_B, h1_C, h1_G] = [None, None, None, None, None]

heuristic_1 = {'G': {}}
heuristic_1['G']['S'] = h1_S
heuristic_1['G']['A'] = h1_A
heuristic_1['G']['B'] = h1_B
heuristic_1['G']['C'] = h1_C
heuristic_1['G']['G'] = h1_G


# heuristic_2: admissible but NOT consistent

[h2_S, h2_A, h2_B, h2_C, h2_G] = [None, None, None, None, None]

heuristic_2 = {'G': {}}
heuristic_2['G']['S'] = h2_S
heuristic_2['G']['A'] = h2_A
heuristic_2['G']['B'] = h2_B
heuristic_2['G']['C'] = h2_C
heuristic_2['G']['G'] = h2_G


# heuristic_3: admissible but A* returns non-optimal path to G

[h3_S, h3_A, h3_B, h3_C, h3_G] = [None, None, None, None, None]

heuristic_3 = {'G': {}}
heuristic_3['G']['S'] = h3_S
heuristic_3['G']['A'] = h3_A
heuristic_3['G']['B'] = h3_B
heuristic_3['G']['C'] = h3_C
heuristic_3['G']['G'] = h3_G


# heuristic_4: admissible but not consistent, yet A* finds optimal path

[h4_S, h4_A, h4_B, h4_C, h4_G] = [None, None, None, None, None]

heuristic_4 = {'G': {}}
heuristic_4['G']['S'] = h4_S
heuristic_4['G']['A'] = h4_A
heuristic_4['G']['B'] = h4_B
heuristic_4['G']['C'] = h4_C
heuristic_4['G']['G'] = h4_G


##### PART 5: Multiple Choice ##################################################

ANSWER_1 = '2'

ANSWER_2 = '4'

ANSWER_3 = '1'

ANSWER_4 = '3'


#### SURVEY ####################################################################

NAME = "Nabib Ahmed"
COLLABORATORS = "Individual Submission"
HOW_MANY_HOURS_THIS_LAB_TOOK = "10"
WHAT_I_FOUND_INTERESTING = "Learned about search implementation"
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = "Clearer explanation for part 3"



###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

# The following lines are used in the online tester. DO NOT CHANGE!

generic_dfs_sort_new_paths_fn = generic_dfs[0]
generic_bfs_sort_new_paths_fn = generic_bfs[0]
generic_hill_climbing_sort_new_paths_fn = generic_hill_climbing[0]
generic_best_first_sort_new_paths_fn = generic_best_first[0]
generic_branch_and_bound_sort_new_paths_fn = generic_branch_and_bound[0]
generic_branch_and_bound_with_heuristic_sort_new_paths_fn = generic_branch_and_bound_with_heuristic[0]
generic_branch_and_bound_with_extended_set_sort_new_paths_fn = generic_branch_and_bound_with_extended_set[0]
generic_a_star_sort_new_paths_fn = generic_a_star[0]

generic_dfs_sort_agenda_fn = generic_dfs[2]
generic_bfs_sort_agenda_fn = generic_bfs[2]
generic_hill_climbing_sort_agenda_fn = generic_hill_climbing[2]
generic_best_first_sort_agenda_fn = generic_best_first[2]
generic_branch_and_bound_sort_agenda_fn = generic_branch_and_bound[2]
generic_branch_and_bound_with_heuristic_sort_agenda_fn = generic_branch_and_bound_with_heuristic[2]
generic_branch_and_bound_with_extended_set_sort_agenda_fn = generic_branch_and_bound_with_extended_set[2]
generic_a_star_sort_agenda_fn = generic_a_star[2]

# Creates the beam search using generic beam args, for optional beam tests
beam = generic_search(*generic_beam) if TEST_GENERIC_BEAM else None

# Creates the A* algorithm for use in testing the optional heuristics
if TEST_HEURISTICS:
    a_star = generic_search(*generic_a_star)
