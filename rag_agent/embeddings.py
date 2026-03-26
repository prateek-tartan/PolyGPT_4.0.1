from __future__ import annotations

import boto3

from .config import Settings


def build_bedrock_embeddings(settings: Settings):
    from langchain_aws import BedrockEmbeddings

    boto3_client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
    )
    return BedrockEmbeddings(
        client=boto3_client,
        model_id=settings.poly_gpt_embedding_model_id,
    )
