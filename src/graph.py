"""
@brief  Module to extend the networkx undirected graph with methods that are
        useful for tooltip detection.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   16 Jan 2021.
"""

import networkx as nx
import numpy as np
import cv2
import random
import math

class ToolGraph(nx.classes.graph.Graph):
    """
    @class This class extends the networkx class with tooltip detection
           capabilities.
    """

    def from_sknw(self, graph):
        """@returns self constructed from an sknw graph."""
        # Remove all nodes and edges
        self.clear()

        # Copy the nodes
        for v in graph.nodes():
            coord = np.round(graph.nodes[v]['o']).astype(np.int)
            self.add_node(v, x=int(coord[1]), y=int(coord[0]), entry=False)

        # Copy the edges
        for e in graph.edges():
            if e[0] != e[1]:
                self.add_coord_edge(e[0], e[1], labels=set())
                # 'labels' will contain the output of the dot product 
                # traversal 
        return self

    def add_coord_edge(self, v, w, labels):
        """@brief Adds an edge and sets the Euclidean distance as weight."""
        dist = np.linalg.norm(np.array(self.coord(v)) \
            - np.array(self.coord(w)), ord=2)
        self.add_edge(v, w, labels=labels, weight=dist)  

    def add_coord_node(self, x, y, entry=False):
        """
        @brief Inserts a new node in the graph and adds a 2D point with 
               coordinates (x, y) as an attribute. The idea of this method is
               to respect the way the sknw module employs to set the x-y
               coordinates of the graph nodes.
        @param[in]       x      x image coordinate.
        @param[in]       y      y image coordinate.
        @returns the number of the new node.
        """
        new_node = max(self.nodes) + 1 
        assert(new_node not in self)
        self.add_node(new_node, x=x, y=y, entry=entry)
        return new_node
    
    def coord(self, v):
        return (self.nodes[v]['x'], self.nodes[v]['y'])

    def neighbour_nodes(self, v):
        """
        @returns a list with the neighbouring nodes of v.
        """
        return [w for w in self.adj[v] if w != v]

    def leaf_nodes(self, coord=False):
        """
        @returns a list of leaf nodes, i.e. all the nodes that have
                 only one neighbour and are not marked as entry.
        """
        if coord:
            return [self.coord(v) for v in self.nodes() if self.is_leaf(v)]
        else:
            return [v for v in self.nodes() if self.is_leaf(v)]

    def entry_nodes(self, coord=False):
        """
        @returns a list of the entry nodes, i.e. all the nodes that have the
                 attribute 'entry=True'.      
        """
        if coord:
            return [self.coord(v) for v in self.nodes() if self.is_entry(v)]
        else:
            return [v for v in self.nodes() if self.is_entry(v)]

    def contract_nodes(self, dst, src_list=[]):  
        """
        @brief Delete all nodes in 'src_list' and connect all the edges
               that were connected to them to the node 'dst'.
        @returns nothing.
        """
        for v in src_list:
            nx.contracted_nodes(self, dst, v, copy=False)
            # Sometimes nx creates unnecessary self loops...
            if dst in self.adj[dst]:
                self.remove_edge(dst, dst)

    def nodes_disconnected_from_entry(self):
        """
        @returns a list of nodes that are not connected to any entry node.
        """
        reachable_nodes = []
        for v in self.entry_nodes():
            reachable_nodes += list(nx.dfs_postorder_nodes(self, source=v))
        return [v for v in self.nodes() if v not in reachable_nodes]

    def prune_disconnected_nodes(self): 
        """
        @brief Deletes nodes that are not connected to an entry node and
               entry nodes that are not connected to anyone.
        """
        # Remove nodes that are not reachable from an entry node
        for v in self.nodes_disconnected_from_entry():
            self.remove_node(v)
        
        # Remove entry nodes that are not connected to anyone
        entry_nodes = self.entry_nodes()
        for v in entry_nodes:
            if len(self.neighbour_nodes(v)) == 0:
                self.remove_node(v)

    def _next_dot_traversal_nodes(self, vi, vj):
        """
        @brief Computes the next graph node to be traversed according to the
               dot product criteria.
        @param[in]  vi  Previously traversed node.
        @param[in]  vj  Currently traversed node.
        """
        solution = self.neighbour_nodes(vj)

        if vi is not None:
            # Get the current direction vector
            diff = np.array(self.coord(vj)) - np.array(self.coord(vi))
            curr_dir_vector =  diff / np.linalg.norm(diff, ord=2) 
            
            # Compute the dot product for all the neighbours
            scores = []
            for v in solution:
                diff = np.array(self.coord(v)) - np.array(self.coord(vj))
                dir_vector = diff / np.linalg.norm(diff, ord=2)  
                scores.append(np.dot(curr_dir_vector, dir_vector))
            
            # Re-order solution according to the scores
            solution = [v for _, v in sorted(zip(scores, solution),
                reverse=True)]

        return solution
    
    def _simplify_path(self, path):
        """
        @brief Given a path starting from node v and ending in node w, this
               method checks whether all the nodes are actually necessary.
               This is because when traversing the graph we can enter in other
               branches and then come back to the 'main' branch and we are
               only interested in the minimal path from a leaf to an entry 
               node.
        @param[in]  path  List of nodes.
        @returns the same or a simpler version of the path.
        """
        return nx.shortest_path(self.subgraph(path), path[0], path[-1])

    def _next_dot_traversal_angles(self, vi, vj):
        """
        @brief Computes all direction changes from 'v' to all its neighbours
               taking into account that the previous node was 'prev_v'.
        @param[in]  vi  Previous node.
        @param[in]  vj  Current node.
        @returns a list of angle changes in degrees.  
        """
        assert(vi is not None)
        assert(vj is not None)
        angles = []
        
        # Get the current direction vector
        diff = np.array(self.coord(vj)) - np.array(self.coord(vi))
        curr_dir_vector =  diff / np.linalg.norm(diff, ord=2) 

        # Compute the dot product for all the neighbours that are 'forward'
        angles = []
        for v in self.neighbour_nodes(vj):
            diff = np.array(self.coord(v)) - np.array(self.coord(vj))
            dir_vector = diff / np.linalg.norm(diff, ord=2)  
            dp = np.clip(np.dot(curr_dir_vector, dir_vector), -1.0, 1.0)
            angles.append(math.degrees(np.arccos(dp)))
        return angles

    def _stop_condition(self, v, visited, stop_condition, ang_thresh=45):
        """
        @brief Evaluates if the stop condition of the traversal has been met.
        @param[in]  v               Current graph node.
        @param[in]  visited         List of previously visited nodes.
        @param[in]  stop_condition  String with the name of the stop
                                    condition you want to use.
        @param[in]  ang_thresh      In case you want to use the
                                    'leaf_or_dir_change' condition this is
                                    the maximum allowed direction change in
                                    degrees. Default is 45 degrees.
        @returns True if the stop condition is reached. Otherwise, False.
        """
        stop_condition_reached = False

        if stop_condition == 'entry':
            # Stop if we reached an entry node
            stop_condition_reached = self.nodes[v]['entry']
        elif stop_condition == 'leaf_or_dir_change':
            # Stop if:
            #    1) Leaf node reached
            #    2) A strong change of direction is needed
            if v in self.leaf_nodes():
                stop_condition_reached = True

            # Stop if we cannot continue in a -sort of- straight line
            prev_v = visited[-1] if visited else None
            if prev_v is not None:
                angles = self._next_dot_traversal_angles(prev_v, v)
                acceptable_angles = [x for x in angles if x < ang_thresh]
                if len(acceptable_angles) == 0:
                    stop_condition_reached = True
        elif stop_condition == 'dir_change':
            # Stop if we cannot continue in a -sort of- straight line
            prev_v = visited[-1] if visited else None
            if prev_v is not None:
                angles = self._next_dot_traversal_angles(prev_v, v)
                acceptable_angles = [x for x in angles if x < ang_thresh]
                if len(acceptable_angles) == 0:
                    stop_condition_reached = True
        else:
            raise ValueError('[ERROR] Unknown stop condition.')

        return stop_condition_reached

    def _recursive_traverse(self, v, visited, stop_condition):
        """
        @brief Greedy dot product traversal. We move first to the node with
               the least change of direction.
        @param[in]       v               Current node.
        @param[in, out]  visited         Previously visited nodes.
        @param[in]       stop_condition  String with the name of the stop 
                                         condition. Options:
                                         'entry' or 'leaf_or_dir_change'.
        @returns True if the stop condition was reached. Otherwise, False.
        """
        # Check stop condition
        stop_condition_reached = self._stop_condition(v, visited, 
            stop_condition)

        # Get the previous node, necessary to compute the direction vector
        prev_v = visited[-1] if visited else None

        # Mark current node as visited
        visited.append(v) 

        # Traverse to those neighbours that have not been already visited
        for w in self._next_dot_traversal_nodes(prev_v, v):
            if w not in visited and not stop_condition_reached:
                stop_condition_reached = self._recursive_traverse(w, visited,
                    stop_condition)
        return stop_condition_reached

    def _label_path_from_leaf_to_entry(self, path):
        """
        @brief Label all the edges on the way from the leaf to the entry node
               with the number of the entry node reached.
        @param[in]  path  List of nodes from leaf to entry node.
        @returns nothing.
        """
        for i in range(len(path) - 1):
            self.edges[path[i], path[i + 1]]['labels'].add(path[-1])

    def _label_edges_connected_to_entry_nodes(self):
        """
        @brief All the edges connected to entry nodes get labelled with the node id of
               such entry node.
        @returns nothing.
        """
        for v in self.entry_nodes():
            for w in self.neighbour_nodes(v):
                self.edges[v, w]['labels'].add(v)

    def dot_product_traversal(self, entry_to_leaf=True, leaf_to_entry=True, 
            next_to_entry=False):
        """
        @brief Traverse from every leaf node to an entry node, labelling the
               edges on the way.
        @param[in]  entry_to_leaf  If True, a dot product traversal is performed
                                   starting from each entry node and finishing when 
                                   a direction change of more than -typically- 45 degrees 
                                   has to be performed (see the documentation of the 
                                   _stop_condition method for the exact angle).
                                   For example, if a leaf is reached, we have to go back, 
                                   this is a 180 direction change, which means that we
                                   have to stop when we reach a leaf.
                    leaf_to_entry  If True, a dot product traversal is performed starting
                                   from each leaf node and stopping when an entry node is
                                   reached.
                    next_to_entry  If True, the edges connected to the entry nodes are 
                                   labelled automatically. This is not necessary if 
                                   entry_to_leaf=True, as the adjacent edges to the entry
                                   node will already be labelled during the traversal.
        @returns nothing but it labels all the edges of the graph with a set of numbers
                 that correspond to the instrument number they are assigned to. As each
                 entry node is assumed to be an instrument the labels assigned to the 
                 edges are the node identifiers of the entrypoints.
        """
        # Label the edges connected to the entrypoints
        if next_to_entry: 
            self._label_edges_connected_to_entry_nodes()

        # Dot product traversal from entry nodes to leaf nodes 
        if entry_to_leaf:
            # Dot product traversal from entry nodes to leaf nodes
            for entry in self.entry_nodes():
                visited = []
                if self._recursive_traverse(entry, visited, 
                        stop_condition='dir_change'):
                    visited = self._simplify_path(visited)

                    # Label the edges on the way with the id of the entry node
                    self._label_path_from_leaf_to_entry(visited[::-1])
        
        # Dot product traversal from leaf nodes to entry nodes
        if leaf_to_entry:
            # Dot product traversal from leaf nodes to entry nodes
            for leaf in self.leaf_nodes():
                visited = []
                if self._recursive_traverse(leaf, visited, 
                        stop_condition='entry'):
                    visited = self._simplify_path(visited) 

                    # Label the edges on the way with the id of the entry node
                    self._label_path_from_leaf_to_entry(visited)
                else:
                    raise RuntimeError("""[ERROR] When traversing from a leaf 
                        we must be able to reach an entry node. This has not 
                        happened, there is a bug in the code.""")

    def _disentangle_nodes(self, nodes):
        """
        @brief Separate nodes that belong to different tools.
        @param[in]  nodes  List of nodes that belong to several instruments.
        @returns nothing.
        """
        for v in nodes:
            x, y = self.coord(v)
            
            # Find out the labels of the edges connected to this node
            labels = set.union(*[self.edges[v, w]['labels'] \
                for w in self.neighbour_nodes(v)])

            for label in labels:
                # Create new node
                new_node = self.add_coord_node(x, y, entry=False)

                # Connect it to all neighbouring nodes of v whose edge is 
                # labelled with the same label
                for w in self.neighbour_nodes(v):
                    if label in self.edges[v, w]['labels']:
                        self.add_coord_edge(new_node, w, labels=set([label]))

        # Delete nodes
        for v in nodes:
            self.remove_node(v)

    def is_entry(self, v):
        return self.nodes[v]['entry']

    def is_leaf(self, v):
        return len(self.neighbour_nodes(v)) == 1 and not self.nodes[v]['entry']

    def overlap_nodes(self):
        """
        @brief An overlap node is a node whose edges are all labelled, but the
               labels are different. This happens, for example, when two 
               tools cross over each other.
        @returns a list of nodes.
        """
        overlap_nodes = []
        candidate_nodes = [v for v in self.nodes() if not self.is_entry(v)]
        for v in candidate_nodes:
            labels = [self.edges[v, w]['labels'] \
                for w in self.neighbour_nodes(v)]
            if labels:
                all_labelled = False not in [bool(s) for s in labels]
                different_labels = len(set.union(*labels)) > 1
                if all_labelled and different_labels:
                    overlap_nodes.append(v)

        return overlap_nodes

    def disentangle_overlap_nodes(self):
        """
        @brief Find and disentangle overlap nodes.
        @returns nothing.
        """
        finished = False
        while not finished:
            overlap_nodes = self.overlap_nodes()
            if overlap_nodes:
                self._disentangle_nodes(overlap_nodes)
            else:
                finished = True

    def unlabelled_edges(self):
        """
        @returns a list of the edges that are not labelled, i.e. they
                 have not been matched to an entry node.
        """
        return [e for e in self.edges() \
            if not self.edges[e[0], e[1]]['labels']]

    def prune_unlabelled_edges(self):
        """@brief Remove edges that are not matched to an entry node."""
        for e in self.unlabelled_edges(): 
            self.remove_edge(e[0], e[1])

    def geodesic_dist(self, v, w):
        """
        @returns the graph-based distance between two nodes.
        """
        try:
            dist = nx.shortest_path_length(self, v, w, weight='weight')
        except nx.NetworkXNoPath:
            dist = None 
        return dist

    def disentangle_tools(self):
        """
        @brief Separate the edges of each tool (or entrypoint) into different graph
               components. This is expected to be executed after the dot product
               traversal, once the edges have been labelled.
        @returns nothing.
        """
        old_nodes = list(self.nodes())

        # For all the tools (tools and entrynodes are synonyms)
        for v in self.entry_nodes():
            # Get the subgraph of this particular tool
            edges = [(u, w) for u, w, data in self.edges(data=True) \
                if v in data['labels']]
            subgraph = self.edge_subgraph(edges)
            
            # Duplicate all the nodes
            old_to_new = {}
            subgraph_nodes = list(subgraph.nodes())
            for u in subgraph_nodes:
                new_node = self.add_coord_node(self.nodes[u]['x'],
                    self.nodes[u]['y'], entry=self.is_entry(u))
                old_to_new[u] = new_node
            
            # Replicate the edges
            for u, w in [(u, w) for u, w in subgraph.edges(data=False)]:
                new_src = old_to_new[u]    
                new_dst = old_to_new[w]
                self.add_coord_edge(new_src, new_dst, labels=set([old_to_new[v]]))

        # Remove old nodes
        for v in old_nodes:
            self.remove_node(v)

    def matched_leaf_nodes(self, v):
        """
        @brief Finds the leaf nodes matched to a particular entry node.
        @param[in]  v  Entry node.
        @returns a list of the matched nodes.
        """
        matched_leaf_nodes = []
        for w in self.leaf_nodes():
            u = self.neighbour_nodes(w)[0]
            if v in self.edges[w, u]['labels'] and nx.has_path(self, v, w):
                matched_leaf_nodes.append(w)
        return matched_leaf_nodes

    def keep_furthest_tips(self, max_tips):
        """
        @brief This method keeps a maximum of 'max_tips' leaf nodes per instrument.
               The rest of the leaf nodes are deleted.
        @returns nothing.
        """
        # For all the tools (tools and entrynodes are synonyms)
        nodes_to_keep = []
        for v in self.entry_nodes():
            # Collect all the leaf nodes associated to this entry node
            associated_leaf_nodes = self.matched_leaf_nodes(v)

            # Compute distances between leaf nodes and this entry node
            dist = {w: self.geodesic_dist(w, v) for w in associated_leaf_nodes}

            # Sort the leaf nodes, furthest first, and keep only max_tips of them
            furthest = [w for w, _ in sorted(dist.items(), reverse=True, 
                key=lambda x: x[1])]
            furthest = furthest[:max_tips]  

            # Get paths to the furthest leaf nodes
            paths = [nx.shortest_path(self, v, w, weight='weight') for w in furthest]

            # Add paths to the nodes we want to keep
            for path in paths:
                for w in path:
                    nodes_to_keep.append(w)
        
        # Delete the nodes that are not in the chains to the furthest leaf nodes
        nodes = list(self.nodes())
        for v in nodes:
            if v not in nodes_to_keep:
                self.remove_node(v)

    def keep_longest_inst(self, max_inst):
        """
        @brief This method keeps only 'max_inst' instruments in the graph, deleting the
               other instruments. The longest chain of each instrument is used to rank
               them.
        @returns nothing.
        """
        if len(self.entry_nodes()) > max_inst:
            lengths = {}

            # Compute the length of the longest chain of each instrument
            for v in self.entry_nodes():
                # Collect all the leaf nodes associated to this entry node (or tool)
                associated_leaf_nodes = self.matched_leaf_nodes(v)

                # Compute distances between leaf nodes and this entry node
                dist = [self.geodesic_dist(w, v) for w in associated_leaf_nodes]

                # Take the distance of the longest chain
                lengths[v] = max(dist) 
            
            # Find the longest instruments    
            longest = [l for l, _ in sorted(lengths.items(), reverse=True, 
                key=lambda x: x[1])]
            longest = set(longest[:max_inst])

            # Delete all the edges that do not belong to a long instrument
            edges = list(self.edges())
            for v, w in edges:
                labels = self.edges[v, w]['labels']
                if not labels.intersection(longest):
                    self.remove_edge(v, w)
                  
    def prune_leaf_nodes(self, max_leaf_nodes_per_entry=2):
        """
        @brief Keep a maximum of leaf nodes per entry node.
        """
        for v in self.entry_nodes():
            associated_leaf_nodes = []

            # Collect all the leaf nodes associated to this entry node
            for w in self.leaf_nodes():
                u = self.neighbour_nodes(w)[0]
                if v in self.edges[w, u]['labels']:
                    associated_leaf_nodes.append(w)
            
            # Compute distances between leaf nodes and this entry node
            dist = [self.geodesic_dist(w, v) for w in associated_leaf_nodes]
            
            # Clean Nones from the distance list
            clean_nodes = []
            clean_dist = []
            for w, d in zip(associated_leaf_nodes, dist):
                if d:
                    clean_nodes.append(w)
                    clean_dist.append(d)
            
            # Sort associated nodes by distance: furthest to closest
            clean_nodes = [x for _, x in sorted(zip(clean_dist, clean_nodes), 
                reverse=True)]
            
            # Remove those leaf nodes that are not top in the ranking
            for w in clean_nodes[max_leaf_nodes_per_entry:]:
                self.remove_node(w)
    
    def draw_onto_image(self, im, node_radius=2,
            node_colour=(0, 255, 255), node_filled=True, 
            line_colour=(0, 255, 255), line_thickness=2,
            entry_colour=(0, 255, 0), leaf_colour=(255, 0, 255), 
            font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, put_text=False):
        """
        @brief Display the graph over the image.
        @param[in]  im           BGR image as a numpy.ndarray.
        @param[in]  graph        networkx.classes.graph.Graph representing 
                                 the skeleton of the tools.
        @param[in]  node_radius  Radius of the nodes in number of pixels. 
        @param[in]  node_colour  BGR colour. 
        @returns an image of the graph drawn on top of the image.
        """
        im_graph = im.copy()

        # Create list of colours, one per entry node
        tool_colour = {}
        for v in self.entry_nodes():
            rand_color = (random.randint(0, 255), random.randint(0, 255), 
                random.randint(0, 255))
            tool_colour[v] = rand_color

        # Change boolean into OpenCV border width
        node_filled = -1 if node_filled else 1
        
        # Add all the edges to the image
        for e in self.edges():
            labels = self.edges[e[0], e[1]]['labels']
            if labels:
                v = labels.pop()
                labels.add(v)
                lc = tool_colour[v] 
            else:
                lc = line_colour
            cv2.line(im_graph, self.coord(e[0]), self.coord(e[1]), lc, 
                line_thickness)

        # Add all the nodes to the image
        for v in self.nodes():
            if v in self.entry_nodes():
                colour = entry_colour 
            elif v in self.leaf_nodes():
                colour = leaf_colour
            else:
                colour = node_colour
            cv2.circle(im_graph, self.coord(v), node_radius, colour, 
                node_filled)
            if put_text:
                cv2.putText(im_graph, str(v), self.coord(v), 
                    font, font_scale, line_colour)

        return im_graph
    
    def draw_onto_canvas(self):
        """@returns an image with the graph plotted on a canvas."""
        import tempfile
        import matplotlib.pyplot as plt
        import os

        # Clear matplotlib plot
        plt.clf()

        # Design node labels
        node_colours = []
        node_labels = {}
        for v in self.nodes():
            colour = None
            label = str(v)
            if self.is_entry(v):
                colour = 'green' 
            elif self.is_leaf(v):
                colour = 'magenta'
            else:
                colour = 'white'
            node_colours.append(colour)
            node_labels[v] = label

        # Design edge labels
        edge_labels = {}
        for v, w, data in self.edges(data=True):
            label = str(int(data['weight'])) + '['
            for ep in data['labels']:
                label += str(ep) + ', '
            if label[-2] == ',':
                label = label[:-2]
            label += ']'
            edge_labels[(v, w)] = label

        # Draw graph
        pos = nx.planar_layout(self)
        nx.draw(self, pos, labels=node_labels, node_color=node_colours)
        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels)

        # Get an image of the graph
        temp_path = os.path.join(tempfile.gettempdir(), '.ToolGraph.png')
        plt.savefig(temp_path)
        im = cv2.imread(temp_path)

        return im

if __name__ == "__main__":
    raise RuntimeError('[ERROR] This module is not a script.')
