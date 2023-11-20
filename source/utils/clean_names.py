import os
import sys

# example_path = "/Users/gregruyoga/Code/gmoneycodes/scrape_me_plate/data/imgs/yolov8"


def clean_names(path):
    """
    Bulk clean names in ./data/imgs/ folder.
    Ensures that the images and labels are in the following format
    Images - {id}.jpg
    Labels - {id}.txt
    :param path: parent directory of images and labels folder
    """
    splits = ['train', 'test', 'valid']
    kinds = ['images', 'labels']

    for split in splits:
        for kind in kinds:
            base = f"{path}/{split}/{kind}"
            ext = "jpg" if kind == "images" else "txt"
            for filename in os.listdir(base):
                if filename.endswith(f".{ext}"):
                    parts = filename.split(".")
                    if len(parts) >= 4:
                        new_name = f"{parts[2]}.{ext}"
                        old_file = os.path.join(base, filename)
                        new_file = os.path.join(base, new_name)

                        os.rename(old_file, new_file)
                        print(f"Renamed {filename} to {new_name}")
                    else:
                        print(f"Skipping {filename} cus its not in a valid format")
                else:
                    print(f"Skipping {filename} cus its not .{ext}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path_arg = sys.argv[1]
        clean_names(path_arg)
    else:
        print("Please provide the path as a command line argument.")
