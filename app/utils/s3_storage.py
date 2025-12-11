"""
S3 Storage Helper for UGC Video Platform

Handles uploading and downloading of generated assets to AWS S3 with proper folder structure:
- person-images/{user_id}/{generation_id}.png
- composites/{user_id}/{generation_id}.png
- videos/{user_id}/{generation_id}.mp4
- products/{user_id}/{product_id}.png
"""

import boto3
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any
import logging
import io
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class S3StorageClient:
    """Client for managing S3 uploads and downloads for video generation assets"""

    def __init__(self):
        """Initialize S3 client with credentials from settings"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.AWS_S3_BUCKET
        self.region = settings.AWS_REGION

    def _get_s3_key(self, asset_type: str, user_id: str, filename: str) -> str:
        """
        Generate S3 key based on asset type and user ID

        Args:
            asset_type: "person-images", "composites", "videos", or "products"
            user_id: User UUID
            filename: File name (e.g., "abc-123.png")

        Returns:
            S3 key path (e.g., "person-images/user-123/abc-123.png")
        """
        return f"{asset_type}/{user_id}/{filename}"

    async def upload_from_url(
        self,
        url: str,
        asset_type: str,
        user_id: str,
        filename: str,
        content_type: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Download file from URL and upload to S3

        Args:
            url: Source URL to download from
            asset_type: "person-images", "composites", "videos", or "products"
            user_id: User UUID
            filename: Filename for S3 storage
            content_type: MIME type (e.g., "image/png", "video/mp4")

        Returns:
            {
                "s3_url": "https://s3.../person-images/user-123/abc-123.png",
                "s3_key": "person-images/user-123/abc-123.png"
            }

        Raises:
            Exception: If download or upload fails
        """
        try:
            # Download file from URL
            logger.info(f"📥 Downloading from {url}")
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                file_data = response.content

            # Infer content type if not provided
            if not content_type:
                if filename.endswith('.mp4'):
                    content_type = 'video/mp4'
                elif filename.endswith('.png'):
                    content_type = 'image/png'
                elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    content_type = 'image/jpeg'
                else:
                    content_type = 'application/octet-stream'

            # Generate S3 key
            s3_key = self._get_s3_key(asset_type, user_id, filename)

            # Upload to S3 with public-read ACL so AI providers can access it
            logger.info(f"☁️  Uploading to S3: {s3_key}")
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=file_data,
                ContentType=content_type,
                ACL='public-read'  # Allow AI providers to download
            )

            # Generate S3 URL
            s3_url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"

            logger.info(f"✅ Upload successful: {s3_url}")

            return {
                "s3_url": s3_url,
                "s3_key": s3_key
            }

        except httpx.HTTPError as e:
            logger.error(f"❌ Failed to download from {url}: {str(e)}")
            raise Exception(f"Download failed: {str(e)}")

        except ClientError as e:
            logger.error(f"❌ S3 upload failed: {str(e)}")
            raise Exception(f"S3 upload failed: {str(e)}")

    async def upload_from_bytes(
        self,
        file_data: bytes,
        asset_type: str,
        user_id: str,
        filename: str,
        content_type: str = 'application/octet-stream'
    ) -> Dict[str, str]:
        """
        Upload file data (bytes) directly to S3

        Args:
            file_data: File content as bytes
            asset_type: "person-images", "composites", "videos", or "products"
            user_id: User UUID
            filename: Filename for S3 storage
            content_type: MIME type

        Returns:
            {
                "s3_url": "https://s3.../person-images/user-123/abc-123.png",
                "s3_key": "person-images/user-123/abc-123.png"
            }
        """
        try:
            s3_key = self._get_s3_key(asset_type, user_id, filename)

            logger.info(f"☁️  Uploading to S3: {s3_key}")
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=file_data,
                ContentType=content_type,
                ACL='public-read'  # Allow AI providers to download
            )

            s3_url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"

            logger.info(f"✅ Upload successful: {s3_url}")

            return {
                "s3_url": s3_url,
                "s3_key": s3_key
            }

        except ClientError as e:
            logger.error(f"❌ S3 upload failed: {str(e)}")
            raise Exception(f"S3 upload failed: {str(e)}")

    def get_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate presigned URL for temporary access to S3 object

        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL string
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url

        except ClientError as e:
            logger.error(f"❌ Failed to generate presigned URL: {str(e)}")
            raise Exception(f"Presigned URL generation failed: {str(e)}")

    async def download_to_bytes(self, s3_key: str) -> bytes:
        """
        Download S3 object to bytes

        Args:
            s3_key: S3 object key

        Returns:
            File content as bytes
        """
        try:
            logger.info(f"📥 Downloading from S3: {s3_key}")
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=s3_key
            )
            file_data = response['Body'].read()

            logger.info(f"✅ Download successful: {len(file_data)} bytes")
            return file_data

        except ClientError as e:
            logger.error(f"❌ S3 download failed: {str(e)}")
            raise Exception(f"S3 download failed: {str(e)}")

    def delete_object(self, s3_key: str) -> bool:
        """
        Delete object from S3

        Args:
            s3_key: S3 object key

        Returns:
            True if successful
        """
        try:
            logger.info(f"🗑️  Deleting from S3: {s3_key}")
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=s3_key
            )

            logger.info(f"✅ Delete successful")
            return True

        except ClientError as e:
            logger.error(f"❌ S3 delete failed: {str(e)}")
            raise Exception(f"S3 delete failed: {str(e)}")

    def list_user_assets(
        self,
        user_id: str,
        asset_type: Optional[str] = None
    ) -> list[str]:
        """
        List all S3 objects for a user

        Args:
            user_id: User UUID
            asset_type: Optional filter by asset type

        Returns:
            List of S3 keys
        """
        try:
            prefix = f"{asset_type}/{user_id}/" if asset_type else f"{user_id}/"

            logger.info(f"📋 Listing S3 objects with prefix: {prefix}")

            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return []

            keys = [obj['Key'] for obj in response['Contents']]

            logger.info(f"✅ Found {len(keys)} objects")
            return keys

        except ClientError as e:
            logger.error(f"❌ S3 list failed: {str(e)}")
            raise Exception(f"S3 list failed: {str(e)}")


# Singleton instance
s3_storage = S3StorageClient()


# Convenience functions for specific asset types

async def upload_person_image(url: str, user_id: str, generation_id: str) -> Dict[str, str]:
    """Upload person image from provider URL to S3"""
    result = await s3_storage.upload_from_url(
        url=url,
        asset_type="person-images",
        user_id=user_id,
        filename=f"{generation_id}.png",
        content_type="image/png"
    )
    # Generate presigned URL for AI providers to access (valid for 1 hour)
    presigned_url = s3_storage.get_presigned_url(result["s3_key"], expiration=3600)
    result["presigned_url"] = presigned_url
    # Use presigned URL for s3_url so AI providers can access it
    result["s3_url"] = presigned_url
    return result


async def upload_composite_image(url: str, user_id: str, generation_id: str) -> Dict[str, str]:
    """Upload composite image from provider URL to S3"""
    result = await s3_storage.upload_from_url(
        url=url,
        asset_type="composites",
        user_id=user_id,
        filename=f"{generation_id}.png",
        content_type="image/png"
    )
    # Generate presigned URL for AI providers to access (valid for 1 hour)
    presigned_url = s3_storage.get_presigned_url(result["s3_key"], expiration=3600)
    result["presigned_url"] = presigned_url
    # Use presigned URL for s3_url so AI providers can access it
    result["s3_url"] = presigned_url
    return result


async def upload_video(url: str, user_id: str, generation_id: str) -> Dict[str, str]:
    """Upload video from provider URL to S3"""
    return await s3_storage.upload_from_url(
        url=url,
        asset_type="videos",
        user_id=user_id,
        filename=f"{generation_id}.mp4",
        content_type="video/mp4"
    )


async def upload_product_image(url: str, user_id: str, product_id: str) -> Dict[str, str]:
    """Upload product image from URL to S3"""
    return await s3_storage.upload_from_url(
        url=url,
        asset_type="products",
        user_id=user_id,
        filename=f"{product_id}.png",
        content_type="image/png"
    )
