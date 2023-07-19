import unittest
import pathlib
import json

from metadata_checker import check_metadata

def recursively_check_metadata(current_path: pathlib.Path):
    for child_path in current_path.glob("*"):
        if child_path.is_dir():
            recursively_check_metadata(child_path)
        else:
            with open(child_path, "rb") as f:  # will close() when we leave this block
                metadata_dict = json.load(f)
            print(f"Checking {child_path}")
            check_metadata(metadata_dict, child_path)

class TestMetadata(unittest.TestCase):
    def test_metadata_files_up_to_date(self):
        path_to_metadata = pathlib.Path("metadata_folder")
        recursively_check_metadata(path_to_metadata)


if __name__ == "__main__":
    unittest.main()
