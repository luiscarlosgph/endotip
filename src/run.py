"""
@brief   Script to run the tooltip detection on a folder containing pairs of 
         image + segmentation mask.
@author  Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date    16 Jan 2021.
"""

import os
import argparse
import ntpath
import cv2
import json

# My imports
import endotip

def parse_command_line_parameters(parser):      
    """
    @brief  Parses the command line parameters provided by the user and makes 
            sure that mandatory parameters are present.
    @param[in]  parser  argparse.ArgumentParser
    @returns an object with the parsed arguments. 
    """
    msg = {
        '--input-dir':  'Path to the input folder.',
        '--output-dir': 'Path to the output folder.',
        '--im-ext':     """Extension of the image files inside the input 
                           folder. Typically '.jpg'""",
        '--seg-suffix': """Suffix of the segmentation files. For example, if 
                           an input image is called image.jpg, and the 
                           corresponding segmentation is image_seg.png,
                           then the suffix is '_seg'.""",
        '--seg-ext':    """Extension of the segmentation mask files. 
                           Typically '.png'""",
        '--max-inst':  'Maximum number of instruments present in the image.',
        '--max-tips':   'Maximum number of instruments present in the image.',
    }
    parser.add_argument('--input-dir', required=True, help=msg['--input-dir'])
    parser.add_argument('--output-dir', required=True, help=msg['--output-dir'])
    parser.add_argument('--im-ext', required=False, default='.jpg', help=msg['--im-ext'])
    parser.add_argument('--seg-suffix', required=False, default='_seg', 
        help=msg['--seg-suffix'])
    parser.add_argument('--seg-ext', required=False, default='.png', help=msg['--seg-ext'])
    parser.add_argument('--max-inst', required=True, help=msg['--max-inst'])
    parser.add_argument('--max-tips', required=True, help=msg['--max-tips'])
    
    args = parser.parse_args()
    args.max_inst = int(args.max_inst)
    args.max_tips = int(args.max_tips)
    return args


def validate_cmd_param(args):
    """
    @brief  The purpose of this function is to assert that the parameters 
            passed in the command line are ok.
    @param[in]  args  Parsed command line parameters.
    @returns nothing.
    """
    if not os.path.isdir(args.input_dir):
        raise ValueError('[ERROR] The input folder does not exist.')
    if os.path.isdir(args.output_dir):
        raise ValueError('[ERROR] The output folder already exists.')


def gather_input_pairs(input_dir, im_ext, seg_suffix, seg_ext):
    """
    @brief  Gather the contents of the input dirctory provided.
    @param[in]  input_dir   Path to the input directory.
    @param[in]  im_ext      Extension of the image files, usually '.jpg'.
    @param[in]  seg_suffix  Suffix of the segmentation files, usually '_seg'.
    @returns two lists with paths to images and their corresponding 
                segmentation masks.
    """
    images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) \
        if im_ext in f]
    masks = []
    for f in images:
        fname, ext = os.path.splitext(ntpath.basename(f))
        mask_path = os.path.join(input_dir, fname + seg_suffix + seg_ext)
        masks.append(mask_path)
    return images, masks


def main(): 
    # Process command line parameters
    parser = argparse.ArgumentParser()
    args = parse_command_line_parameters(parser)
    validate_cmd_param(args)
    
    # Create output folder
    os.mkdir(args.output_dir)
    
    # Gather the list of input images and their corresponding segmentation 
    # masks
    images, masks = gather_input_pairs(args.input_dir, args.im_ext, 
        args.seg_suffix, args.seg_ext)  
    
    # Loop through the images detecting the tooltips
    tool_detector = endotip.Detector(max_inst=args.max_inst, max_tips=args.max_tips)
    for im_path, mask_path in zip(images, masks):
        print('Processing ' + im_path + ' ...')
        fname, ext = os.path.splitext(ntpath.basename(im_path))

        # Load image and mask
        im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) 

        # Run instrument detector
        tool_detector.detect(im, mask) 

        # Get all the entrynodes
        entrynodes = tool_detector.get_entry_nodes(padding=False)

        # Get all the tips
        tips = tool_detector.get_tips(padding=False)
            
        # Get tips for images with two instruments
        if args.max_inst == 2:
            ltips = tool_detector.get_left_instrument_tips(padding=True)
            rtips = tool_detector.get_right_instrument_tips(padding=True)

            # To be consistent with the keypoint annotation tool in
            #    https://github.com/luiscarlosgph/keypoint-annotation-tool 
            # the format of output is as follows (in this example max_tips=2):
            # 
            # {"tooltips": [
            #     {"x": 263, "y": 168}, # Left tip 
            #     {"x": 586, "y": 301}, # Left tip 
            #     {"x": 581, "y": 210}, # Right tip 
            #     {"x": 500, "y": 116}, # Right tip
            # ]}
            lrtips = {'tooltips': []}
            for tip in ltips:
                lrtips['tooltips'].append(tip)
            for tip in rtips:
                lrtips['tooltips'].append(tip)

            # Generate the paths of the left/right output file
            ltips_path = os.path.join(args.output_dir, fname + '_ltips.json')
            rtips_path = os.path.join(args.output_dir, fname + '_rtips.json')
            lrtips_path = os.path.join(args.output_dir, fname + '_lrtips.json')

            # Save the left/tight tooltips
            with open(ltips_path, 'w') as outfile:
                json.dump(ltips, outfile)
            with open(rtips_path, 'w') as outfile:
                json.dump(rtips, outfile)
            with open(lrtips_path, 'w') as outfile:
                json.dump(lrtips, outfile)
    
        # Generate the paths of the output files 
        tips_path = os.path.join(args.output_dir, fname + '_tips.json')
        entry_nodes_path = os.path.join(args.output_dir, fname + '_entrynodes.json')

        # Save the tooltips in the output folder
        with open(entry_nodes_path, 'w') as outfile:
            json.dump(entrynodes, outfile)
        with open(tips_path, 'w') as outfile:
            json.dump(tips, outfile)

        # Save the output graphs for debugging purposes
        graph_im = tool_detector._graph.draw_onto_canvas()
        graph_im_path = os.path.join(args.output_dir, fname + '_graph_onto_canvas.jpg')
        cv2.imwrite(graph_im_path, graph_im)
         
        graph_onto_im = tool_detector._graph.draw_onto_image(im)
        graph_onto_im_path = os.path.join(args.output_dir, 
            fname + '_graph_onto_image_.jpg')
        cv2.imwrite(graph_onto_im_path, graph_onto_im)


if __name__ == '__main__':
    main()
