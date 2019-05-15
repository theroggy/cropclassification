import os, re

def create_run_dir(class_base_dir: str, 
                   reuse_last_run_dir: bool):
    pattern = re.compile('Run_[0-9]{3}')
    dir_list = [x.path for x in os.scandir(class_base_dir) if x.is_dir() and re.search(pattern, x.path) ]
 
    if (dir_list is None or not any(dir_list)):
        # first run
        return os.path.join(class_base_dir, f"Run_{1:03d}")

    # get last run and increment if needed
    last_dir = sorted(dir_list, reverse=True)[0]

    if (reuse_last_run_dir):
        return last_dir

    last_dir_iteration = re.search(pattern, last_dir) 
    last_iteration = int(last_dir_iteration.group().split('_')[1]) 
        
    return os.path.join(class_base_dir, f"Run_{last_iteration + 1:03d}")