# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    comment: str = Field(
        ...,
        title="YouTube Comment",
        description="The text of the YouTube comment to analyze",
        min_length=1,
    )


class BatchPredictionRequest(BaseModel):
    comments: List[str] = Field(..., title="List of YouTube Comments", min_items=1)
