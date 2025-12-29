import hashlib
import pickle
from logging import Logger
from pathlib import Path

from geotessera import GeoTessera
import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.transform import from_bounds, Affine

from rasterio.warp import reproject, Resampling

from pangaea.encoders.base import Encoder


class TesseraEncoder(Encoder):
    """
    Encoder that uses pre-computed GeoTessera embeddings from:
    https://github.com/ucam-eo/geotessera

    Instead of processing raw satellite images, this downloads embeddings
    from GeoTessera based on bounding box and year metadata, then maps them
    to the target pixel resolution.
    
    Args:
        input_bands: Input bands (not used, but required by base class)
        input_size: Expected input size (used for validation)
        output_dim: Output embedding dimension (128 for GeoTessera)
        output_layers: Output layers (single layer for GeoTessera)
        encoder_weights: Not used for GeoTessera
        download_url: Not used for GeoTessera
        cache_dir: Directory to cache downloaded embeddings
    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        output_dim: int,
        output_layers: list[int],
        encoder_weights: str,
        download_url: str,
        cache_dir: str = "./geotessera_cache",
    ):
        super().__init__(
            model_name="GeoTessera",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=128,
            output_dim=output_dim,
            output_layers=output_layers,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )
        # init the geotessera client
        self.gt = GeoTessera()

        # setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_encoder_weights(self, logger: Logger) -> None:
        # not applicable for GeoTessera
        logger.info(
            f"GeoTessera encoder initialized, will use embeddings from year specified in dataset metadata. "
            f"Cache directory: {self.cache_dir}"
        )

    def _get_cache_key(self, bounds: tuple, year: int, height: int, width: int) -> str:
        # return hash of image bounds, year, height, width
        cache_str = f"{bounds}_{year}_{height}_{width}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        # get path to cached embedding file given key
        return self.cache_dir / f"embedding_{cache_key}.pt"

    def _download_and_map_embedding(
        self,
        bounds: tuple[float, float, float, float],
        year: int,
        target_height: int,
        target_width: int,
        target_crs: str,
        target_transform: tuple,
    ) -> torch.Tensor:
        """
        Download embedding from GeoTessera and map to target pixel grid.

        Args:
            bounds: Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
            year: Year to download embeddings for (from dataset metadata)
            target_height: Target height in pixels
            target_width: Target width in pixels
            target_crs: Target coordinate reference system (e.g., 'EPSG:4326')
            target_transform: Affine transform for target grid (6-element tuple)

        Returns:
            Tensor of shape (128, target_height, target_width)
        """
        print("downloading bounds:", bounds, "year:", year)
        # download embeddings from GeoTessera
        tiles_to_fetch = self.gt.registry.load_blocks_for_region(bounds=bounds, year=year)
        embeddings_generator = self.gt.fetch_embeddings(tiles_to_fetch)

        # create target array for final output
        target_array = np.zeros((128, target_height, target_width), dtype=np.float32)

        # convert target_transform tuple to Affine object
        if isinstance(target_transform, tuple):
            target_transform_affine = Affine(*target_transform)
        else:
            target_transform_affine = target_transform

        # process each tile from the generator
        for tile_year, tile_lon, tile_lat, embedding, tile_crs, tile_transform in embeddings_generator:
            # embedding is already dequantized, shape (H, W, 128)
            # transpose to (128, H, W) for rasterio
            embedding_array = np.transpose(embedding, (2, 0, 1)).astype(np.float32)

            # reproject this tile's embedding to target grid
            # this will accumulate/blend overlapping tiles
            temp_array = np.zeros((128, target_height, target_width), dtype=np.float32)

            reproject(
                source=embedding_array,
                destination=temp_array,
                src_transform=tile_transform,
                src_crs=tile_crs,
                dst_transform=target_transform_affine,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )

            # accumulate tiles (simple average for overlapping regions)
            mask = temp_array != 0  # non-zero values from this tile
            target_array = np.where(mask, temp_array, target_array)

        # convert to torch tensor
        embedding_tensor = torch.from_numpy(target_array).float()

        return embedding_tensor

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass - load or download GeoTessera embeddings.

        Args:
            image: Dictionary containing:
                - "optical": Tensor of shape (B, C, T, H, W) (not used directly)
                - "_metadata": Dictionary with:
                    - "bounds": List of B bounding boxes (min_lon, min_lat, max_lon, max_lat)
                    - "index": List of B dataset indices
                    - "crs": List of B CRS strings
                    - "transform": List of B transform tuples
                    - "year": List of B years (endYear from dataset config)

        Returns:
            List containing single tensor of shape (B, 128, H, W)
        """
        # get batch size and target dimensions from the optical input
        # Handle both 4D (B, C, H, W) and 5D (B, C, T, H, W) inputs
        if image["optical"].ndim == 5:
            B, _, _, H, W = image["optical"].shape
            device = image["optical"].device
        elif image["optical"].ndim == 4:
            B, _, H, W = image["optical"].shape
            device = image["optical"].device
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got {image['optical'].ndim}D")

        # extract metadata, throw if not found
        if "_metadata" not in image:
            raise ValueError(
                "TesseraEncoder requires '_metadata' in image dict. "
                "Ensure dataset.__getitem__ includes bounding box, index, CRS, transform, and year."
            )

        metadata = image["_metadata"]
        bounds_list = metadata["bounds"]
        crs_list = metadata["crs"]
        transform_list = metadata["transform"]
        year_list = metadata["year"]

        # process each sample in the batch
        batch_embeddings = []

        for b in range(B):
            bounds = bounds_list[b]
            crs = crs_list[b]
            transform = transform_list[b]
            year = year_list[b]

            # generate cache key
            cache_key = self._get_cache_key(bounds, year, H, W)
            cache_path = self._get_cache_path(cache_key)

            if cache_path.exists():
                # load from disk cache
                embedding = torch.load(cache_path, map_location='cpu', weights_only=True)
            else:
                # download and map embedding
                embedding = self._download_and_map_embedding(
                    bounds=bounds,
                    year=year,
                    target_height=H,
                    target_width=W,
                    target_crs=crs,
                    target_transform=transform,
                )

                # save to disk cache
                torch.save(embedding, cache_path)

            batch_embeddings.append(embedding)

        # stack batch and move to correct device
        embeddings = torch.stack(batch_embeddings, dim=0).to(device)  # (B, 128, H, W)

        # return as list for compatibility with decoder interface
        return [embeddings]
