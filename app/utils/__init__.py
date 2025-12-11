"""Utility modules for the UGC Video platform"""

from .s3_storage import (
    s3_storage,
    upload_person_image,
    upload_composite_image,
    upload_video,
    upload_product_image,
    S3StorageClient
)

__all__ = [
    's3_storage',
    'upload_person_image',
    'upload_composite_image',
    'upload_video',
    'upload_product_image',
    'S3StorageClient'
]
