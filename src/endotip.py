"""
@brief   Detector of surgical tooltips in endoscopic images.
@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    16 Jan 2021.
"""

import skimage.morphology
import sknw
import numpy as np
import cv2
import sklearn.metrics

# My imports
import endoseg
import endotip.graph

class Detector():
    """
    @class Detector aims to localize the tooltips of the instruments present 
           in endoscopic images.
    """

    def __init__(self, max_inst=2, max_tips=2):
        """
        @param[in]  max_inst  Maximum number of instruments in the image.
        @param[in]  max_tips  Maximum number of tips per instrument.
        """
        self.max_inst = max_inst
        self.max_tips = max_tips
        self._graph = None
        self._im = None
    
    @staticmethod
    def skel(tool_seg):
        """
        @brief Convert the tool segmentation into a pixel-wide skeleton.
        @returns a single-channel image with the pixels of the skeleton
                 labelled as 255 and the background as zero.
        """
        return skimage.morphology.skeletonize_3d(tool_seg.astype(bool))
    
    @staticmethod
    def skel2graph(skel):
        """
        @brief Convert the pixel-wide skeleton into a networkx graph.
        @returns a ToolGraph representing the skeleton
                 provided as input.
        """
        return endotip.graph.ToolGraph().from_sknw(sknw.build_sknw(skel))

    @staticmethod
    def find_entry_pixels(ep_region_instance_seg, endo_seg, 
            border_thickness=1):
        """
        @brief This method receives the connected component segmentation of
               the entrypoints, which usually looks like a curved rectangle.
               The objective is then to reduce the curved rectangle into a
               single pixel that can be used as a tool entrypoint.
        @param[in]  ep_region_instance_seg  Instance segmentation of the 
                                            border entrypoints.
        @param[in]  endo_seg                Segmentation of the visible
                                            endoscopic area.
        @returns a dictionary with all the entrypoint pixels found. The keys 
                 are integers and the values are [x, y] pairs.
        """ 
        # Get a mask of the pixel-wide border
        #contours, hierarchy = cv2.findContours(endo_seg, cv2.RETR_TREE, 
        #    cv2.CHAIN_APPROX_SIMPLE)
        #border_seg = np.zeros_like(endo_seg)
        #cv2.drawContours(border_seg, contours, -1, 255, border_thickness)
        
        # Get the instance segmentation of the pixel-wide border 
        #ep_region_instance_seg = ep_region_instance_seg.astype(np.uint8)
        #border_inst_seg = cv2.bitwise_and(border_seg, ep_region_instance_seg)
        
        # Get the 1-pixel entrypoint instance segmentation
        entry_pixels = {}
        num_ep = np.max(ep_region_instance_seg)
        for i in range(1, num_ep + 1):
            # Bruteforce geometric median
            #points = np.vstack(np.where(border_inst_seg == i)).T
            points = np.vstack(np.where(ep_region_instance_seg == i)).T
            dist = sklearn.metrics.pairwise_distances(points, points)
            geomedian = points[np.argmin(dist.sum(axis=1))]  # [y, x]
            entry_pixels[i] = geomedian[::-1]  # [x, y] 

        return entry_pixels

    def extract_entry_nodes(self, endo_seg, tool_seg, graph, margin=32):
        """
        @brief Detects the entrypoints and fuses them into a single
               entrypoint per instrument. All the entry nodes are marked with
               the attribute 'entry=True'. The rest of the nodes are marked
               with 'entry=False'.
        @param[in]       endo_seg  Endoscopic area segmentation. 
        @param[in]       tool_seg  Binary (0, 255) tool segmentation.
        @param[in, out]  graph     ToolGraph representing 
                                   the skeleton of the tools.
        """

        # Get the segmentation of the endoscopic border
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (margin, margin))
        smaller_endo_seg = cv2.erode(endo_seg, kernel, iterations=1) 
        border_seg = endo_seg - smaller_endo_seg
        
        # Get the segmentation of the entrypoints
        entry_mask = cv2.bitwise_and(tool_seg, border_seg)

        # Find connected components in the entrypoint segmentation
        ret, ep_region_instance_seg = cv2.connectedComponents(entry_mask)

        # Find entrypoint pixels based on the entrypoint segmentation
        entry_pixels = Detector.find_entry_pixels(ep_region_instance_seg, 
            endo_seg)

        # Group nodes in sets, those inside the same entry CC go together 
        entry_nodes = [set() for i in range(len(entry_pixels) + 1)]
        for v in graph.nodes():
            x, y = graph.coord(v)
            entry_nodes[ep_region_instance_seg[y, x]].add(v)

        # Collapse the entry nodes within the same entry region
        for p in entry_pixels:
            # Create new entry node
            new_node = graph.add_coord_node(entry_pixels[p][0],
                entry_pixels[p][1], entry=True)

            # Collapse all the nodes within the entry region into the entry 
            # node
            graph.contract_nodes(new_node, entry_nodes[p])

        # TODO: Special case: all the nodes of the graph are within the 
        # entry connected component, needs to be trated differently

    def detect(self, im, raw_tool_seg):
        """
        @brief Localise the tooltips of the surgical instruments in the image.
        @param[in]  im            BGR image as a numpy.ndarray.
        @param[in]  raw_tool_seg  2D binary mask containing a semantic 
                                  tool-background segmentation. 
        @returns a JSON containing the location of the tooltips in the image.
        """
        # Segment the visible area of the endoscopic image
        endo_segmenter = endoseg.Segmenter()
        endo_seg = endo_segmenter.segment(im, erode_iterations=1)
        
        # We force the tool segmentation to be inside the visible area
        tool_seg = cv2.bitwise_and(raw_tool_seg, endo_seg)

        # Get a skeleton of the tools
        skel = Detector.skel(tool_seg)

        # Convert tool skeleton into a graph
        graph = Detector.skel2graph(skel)
        
        # Instrument entry node extraction
        self.extract_entry_nodes(endo_seg, tool_seg, graph)

        # Remove nodes that are not connected to the extracted entry nodes
        # and entry nodes that are not connected to anyone
        graph.prune_disconnected_nodes()

        # Dot product traversal edge labelling
        graph.dot_product_traversal()

        # Separate different tools into different graph components
        graph.disentangle_tools()

        # Only the furthest tips are kept
        graph.keep_furthest_tips(self.max_tips)

        # Only the instruments with the furthest tips are kept
        graph.keep_longest_inst(self.max_inst)

        # Store image and tool graph
        self._graph = graph
        self._im = im

    def get_tips(self, padding=False):
        """
        @param[in]  padding  If True, the list of tips is padded with Nones until we
                             reach 'max_inst' * 'max_tips'.
        @returns a dictionary with a list of leaf nodes in the following format:
                 {'tips': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]}. 
                 The tips are not in any particular order.
        """
        tips = {'tips': [{'x': int(v[0]), 'y': int(v[1])}
            for v in self._graph.leaf_nodes(coord=True)]}
        
        # Fill the rest with None if the user wants
        if padding:
            remaining = (self.max_inst * self.max_tips) - len(tips['tips'])
            for _ in range(remaining):
                tips['tips'].append(None)
        
        return tips

    def get_entry_nodes(self, padding=False):
        """
        @param[in]  padding  If True, the list of entry nodes will be filled with None
                             values until we reach 'max_inst'.
        @returns a dictionary with a list of entrynodes in the following format: 
                 {'entrynodes': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]}
        """
        entrynodes = {'entrynodes': [{'x': int(v[0]), 'y': int(v[1])} 
            for v in self._graph.entry_nodes(coord=True)]}
        
        # Fill the rest with None values if the user wants
        if padding:
            remaining = self.max_inst - len(entrynodes['entrynodes'])
            for _ in range(remaining):
                entrynodes['entrynodes'].append(None)

        return entrynodes
             
    def get_left_instrument_tips(self, padding=False):
        assert(self.max_inst == 2)
        left_tips = []
        
        # Get entry nodes and their coordinates in the image 
        ep = {v: self._graph.coord(v) for v in self._graph.entry_nodes()}

        # If there are entry nodes, let's find if there is a left instrument
        left_entry_node = None
        if ep:
            keys = list(ep.keys())
            f_x = ep[keys[0]][0]

            # If there is only one instrument 
            if len(ep) == 1:
                if f_x < self._im.shape[1] // 2:
                    left_entry_node = keys[0] 
            else:
                s_x = ep[keys[1]][0]
                if s_x > f_x:
                    left_entry_node = keys[0]
                else:
                    left_entry_node = keys[1]

        # Find the tips associated with the left instrument entry node
        leaf_nodes = self._graph.matched_leaf_nodes(left_entry_node)
        for v in leaf_nodes:
            coord = self._graph.coord(v)
            left_tips.append({'x': coord[0], 'y': coord[1]})

        if padding:
            remaining = self.max_tips - len(left_tips) 
            for _ in range(remaining):
                left_tips.append({'x': None, 'y': None})
        
        return left_tips

    def get_right_instrument_tips(self, padding=False):
        assert(self.max_inst == 2)
        right_tips = []
        
        # Get entry nodes and their coordinates in the image 
        ep = {v: self._graph.coord(v) for v in self._graph.entry_nodes()}

        # If there are entry nodes, let's find if there is a right instrument
        right_entry_node = None
        if ep:
            keys = list(ep.keys())
            f_x = ep[keys[0]][0]

            # If there is only one instrument 
            if len(ep) == 1:
                if f_x >= self._im.shape[1] // 2:
                    right_entry_node = keys[0] 
            else:
                s_x = ep[keys[1]][0]
                if s_x <= f_x:
                    right_entry_node = keys[0]
                else:
                    right_entry_node = keys[1]

        # Find the tips associated with the right instrument entry node
        leaf_nodes = self._graph.matched_leaf_nodes(right_entry_node)
        for v in leaf_nodes:
            coord = self._graph.coord(v)
            right_tips.append({'x': coord[0], 'y': coord[1]})

        if padding:
            remaining = self.max_tips - len(right_tips) 
            for _ in range(remaining):
                right_tips.append({'x': None, 'y': None})
        
        return right_tips

    '''
    def plot(self, **kwargs):
        """@returns the tool graph plotted on top of the input image."""
        if self._graph is None:
            raise RuntimeError("""[ERROR] You want to plot the results of the
                tool detector but you have not called the detect() method.""")
        #return self.graph.draw_onto_image(self.im, **kwargs)
        return self._graph.draw_onto_canvas()
    '''
