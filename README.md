Description
-----------
Given an endoscopic image and a tool-background semantic segmentation, this module detects the tooltips of the instruments.

<!--
Install dependencies
--------------------
TODO

Install with pip
----------------
TODO
-->

Install from source
-------------------
```
$ python3 setup.py install --user
```

Run tooltip detection 
---------------------
The input folder is expected to contain ```*.jpg``` images and ```*.png``` segmentation
masks with suffix ```_seg```, e.g. ```image.jpg``` and ```image_seg.png```. 

To detect the tips of the surgical instuments run:
```
$ python3 -m endotip.run --input-dir <path_to_input_folder> --output-dir <path_to_output_folder> --max-inst 2 --max-tips 2
```
