"""Pydantic schemas for structured outputs in the news agent."""

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """Schema for user content preferences (topics, themes, areas of interest)."""

    user_preferences: str = Field(
        description="User's content preferences as a formatted string with bullet points for topics and areas of interest"
    )


class UserNewsSourcePreferences(BaseModel):
    """Schema for user news source preferences (websites, publications)."""

    user_news_source_preferences: str = Field(
        description="User's news source preferences as a formatted string with bullet points for websites and publication names"
    )
