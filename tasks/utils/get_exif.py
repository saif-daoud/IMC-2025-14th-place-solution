import os
from PIL import Image, ExifTags
import json

def task_get_exif(params):
    if params["pdb"]:
        import pdb;pdb.set_trace()
    
    image_paths = params["data_dict"]
    h_w_exif = {}
    for image_path in image_paths:
        img = Image.open(image_path)
        w, h = img.size
        h_w_exif[os.path.basename(image_path)] = {'h': h, 'w': w, 'exif': img._getexif()}
    
    work_dir = params["work_dir"]
    output_path = os.path.join(work_dir, params["output"])
    with open(output_path, "w") as f:
        json.dump(h_w_exif, f, indent=4)