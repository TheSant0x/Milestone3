from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Intent(BaseModel):
    category: Literal["question", "recommendation", "search"] = Field(
        ..., 
        description="The category of the user's intent. 'question' for specific facts, 'recommendation' for suggestions, 'search' for looking up specific entities."
    )
    reasoning: str = Field(..., description="Brief explanation of why this category was chosen.")

class Entities(BaseModel):
    locations: List[str] = Field(default_factory=list, description="Cities or Countries mentioned.")
    hotels: List[str] = Field(default_factory=list, description="Specific hotel names.")
    traveller_types: List[str] = Field(default_factory=list, description="Type of traveller (e.g. Solo, Family, Couple).")
    attributes: List[str] = Field(default_factory=list, description="Desired attributes like 'pool', 'wifi', 'cheap', 'clean'.")
    dates: List[str] = Field(default_factory=list, description="Dates or duration mentioned.")
