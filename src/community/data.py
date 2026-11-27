from pydantic import BaseModel, Field

class CommunityFinding(BaseModel):
  summary: str = Field(description="Short, descriptive heading for the finding.")
  explanation: str = Field(description="Explanation of a distinct community insight, supported by the supplied data.")

class CommunityReport(BaseModel):
  title: str = Field(description="Short, descriptive title for the community.")
  summary: str = Field(description="Concise overview of the community and its defining characteristics.")
  rating: float = Field(ge=0.0, le=10.0, description="Qualitative assessment of narrative significance.")
  rating_explanation: str = Field(description="Justification of the narrative significance rating.")
  findings: list[CommunityFinding] = Field(min_length=1, max_length=5, description="Distinct, meaningful insights about the community.")


# Returns for global search
class GlobalMapPoint(BaseModel):
  description: str = Field(description="Relevant information answering the user's question, including supporting community references.")
  score: int = Field(ge=0, le=100, description="Relevance of this information to the user's question.")

class GlobalMapResponse(BaseModel):
  points: list[GlobalMapPoint] = Field(description="Relevant facts and insights from the community. Empty if nothing is relevant.")