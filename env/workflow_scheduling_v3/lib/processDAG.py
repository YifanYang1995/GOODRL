import xml.etree.ElementTree as ET
import networkx as nx

def buildGraph(type, filename):
    tot_processTime = 0
    dag = nx.DiGraph(type=type)
    with open(filename, 'rb') as xml_file:
        tree = ET.parse(xml_file)
        xml_file.close()
    root = tree.getroot()
    for child in root:
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}job':
            size = 0
            for p in child:
                size += int(p.attrib['size'])
            dag.add_node(int(child.attrib['id'][2:]), taskSize = float(child.attrib['runtime']) * 16, size=size)
            tot_processTime += float(child.attrib['runtime']) * 16
            # dag.add_node(child.attrib['id'], taskSize = float(child.attrib['runtime'])*16, size=size)
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
            kid = int(child.attrib['ref'][2:])
            for p in child:
                parent = int(p.attrib['ref'][2:])
                dag.add_edge(parent, kid)
    return dag, tot_processTime


def get_longestPath_nodeWeighted(G):
    entrance = [n for n, d in G.in_degree() if d == 0]
    exit = [n for n, d in G.out_degree() if d == 0]
    cost = 0
    for root in entrance:
        for leaf in exit:
            for path in nx.all_simple_paths(G, source=root, target=leaf):  
                temp = sum(G.nodes[node]['taskSize'] for node in path)
                if temp > cost:
                    cost = temp
    return cost

def get_shortest_weighted_path_sum(G):
    entrance = [n for n, d in G.in_degree() if d == 0]
    exit = [n for n, d in G.out_degree() if d == 0]
    cost = 0
    for root in entrance:
        for leaf in exit:
            try:
                shortest_path = nx.shortest_path(G, source=root, target=leaf, weight='taskSize', method='dijkstra')
                path_sum = sum(G.nodes[node]['taskSize'] for node in shortest_path)
                if path_sum > cost:
                    cost = path_sum
            except nx.NetworkXNoPath:
                cost = float('inf')
    return cost


def get_shortest_paths_to_end(G, start):
    """
    Calculate the shortest paths from any node in G to the target node.
    
    :param G: A directed graph (networkx DiGraph)
    :param start: The start node
    :return: A dictionary with nodes as keys and shortest path lengths to the target as values
    """
    shortest_paths = {}
    for node in G.nodes:
        try:
            # Using Dijkstra's algorithm to find the shortest path length from node to target
            path_length = nx.shortest_path_length(G, source=start, target=node, weight='taskSize', method='dijkstra')
            shortest_paths[node] = path_length
        except nx.NetworkXNoPath:
            # If there is no path from the node to the target, set the path length to infinity
            shortest_paths[node] = float('inf')
    return shortest_paths

def compute_initial_earliest_finish_times(G, generateTime):
    # Topological sort
    topo_order = list(nx.topological_sort(G))
    
    # Initialize the earliest end time of all nodes to 0
    earliest_finish_times = {node: generateTime for node in G.nodes}
    
    # Traverse the topologically sorted nodes
    for node in topo_order:
        # The start time of the current node is the maximum value of the earliest end time of all predecessor nodes
        if G.in_edges(node):
            max_predecessor_finish_time = max(earliest_finish_times[predecessor] for predecessor in G.predecessors(node))
        else:
            max_predecessor_finish_time = generateTime
        
        # The earliest end time of the current node = the start time of the current node + the processing time of the current node
        earliest_finish_times[node] = max_predecessor_finish_time + G.nodes[node]['estimatedET']
    
    return earliest_finish_times

def update_earliest_finish_times(G, node, actual_finish_time, earliest_finish_times):
    # Update the earliest end time of the current node to the actual completion time
    if abs(earliest_finish_times[node] - actual_finish_time) > 1e-5:
        earliest_finish_times[node] = actual_finish_time
        
        # Use a queue to perform a breadth-first search to update the earliest end time of all successor nodes
        queue = [node]
        
        while queue:
            current_node = queue.pop(0)
            
            for successor in G.successors(current_node):
                # The start time of the current node is the maximum value of the earliest end time of all predecessor nodes
                max_predecessor_finish_time = max(earliest_finish_times[predecessor] for predecessor in G.predecessors(successor))
                
                # The earliest end time of the current node = the start time of the current node + the processing time of the current node
                new_finish_time = max_predecessor_finish_time + G.nodes[successor]['estimatedET']
                
                if new_finish_time > earliest_finish_times[successor]:
                    earliest_finish_times[successor] = new_finish_time
                    queue.append(successor)
    
    return earliest_finish_times

def update_node_weight_and_compute_finish_times(G, generateTime, node, new_weight, actual_finish_times):
    G.nodes[node]['estimatedET'] = new_weight  # replace weights
    earliest_finish_times = compute_initial_earliest_finish_times(G, generateTime)
    for idx, actual_finish_time in actual_finish_times.items(): 
        earliest_finish_times = update_earliest_finish_times(G, idx, actual_finish_time, earliest_finish_times)

    return earliest_finish_times 
