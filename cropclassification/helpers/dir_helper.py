import os

def create_dir(class_base_dir: str, 
               reuse_last_run_dir: bool):
               
    max_run_dir_id = 998
    prev_class_dir = None
    class_dir = None 
    for i in range(max_run_dir_id):
        # Check if we don't have too many run dirs for creating the dir name
        if i >= max_run_dir_id:
            raise Exception("Please cleanup the run dirs, too many!!!")
            
        # Now search for the last dir that is in use
        class_dir = os.path.join(class_base_dir, f"Run_{i+1:03d}")
        if os.path.exists(class_dir):
            continue
        else:
            # If we want to reuse the last dir, do so...
            if reuse_last_run_dir and prev_class_dir is not None:
                class_dir = prev_class_dir            
            else:
                # Otherwise create new dir name with next index
                class_dir = os.path.join(class_base_dir, f"Run_{i+1:03d}")         
                break 

    return class_dir