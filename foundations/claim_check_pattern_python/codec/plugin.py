import os
from temporalio.client import ClientConfig, SimplePlugin
from temporalio.converter import DataConverter

from .claim_check import ClaimCheckCodec


class ClaimCheckPlugin(SimplePlugin):
    """Temporal plugin that integrates the Claim Check codec with client configuration."""

    def __init__(self):
        """Initialize the plugin with S3 connection configuration."""
        self.bucket_name = os.getenv("S3_BUCKET_NAME", "temporal-claim-check")
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        super().__init__(configure_client=self._configure_client)

    def _configure_client(self, config: ClientConfig) -> ClientConfig:
        """Apply the claim check configuration to the client."""
        # Configure the data converter with claim check codec
        default_converter_class = config["data_converter"].payload_converter_class
        claim_check_codec = ClaimCheckCodec(
            bucket_name=self.bucket_name,
            endpoint_url=self.endpoint_url,
            region_name=self.region_name
        )

        config["data_converter"] = DataConverter(
            payload_converter_class=default_converter_class,
            payload_codec=claim_check_codec
        )
        return config
