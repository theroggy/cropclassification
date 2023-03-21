from typing import List


class ImageProfile:
    def __init__(
        self,
        name: str,
        satellite: str,
        collection: str,
        bands: List[str],
        process_options: dict,
    ):
        self.name = name
        self.satellite = satellite
        self.collection = collection
        self.bands = bands
        self.process_options = process_options
