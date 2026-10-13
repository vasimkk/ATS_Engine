from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

class GetMoreJDsRequest(BaseModel):
    N: int = 10
    LastJDid: str
    top_k: int = 1000
    ResumeID: str

class EnhanceResumeRequest(BaseModel):
    JDid: str
    ResumeID: str
