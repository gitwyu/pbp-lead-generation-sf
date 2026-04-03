from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

class ContactInfo(BaseModel):
  name: str = Field(description="Full name")
  title: str = Field(description="Job title")
  email: str = Field(description="Email address")
  linkedin: str = Field(description="LinkedIn URL")

class EnrichedLeadData(BaseModel):
  contact_1: ContactInfo = Field(description="First contact")
  contact_2: ContactInfo = Field(description="Second contact")
  general_email: str = Field(description="General inbox (e.g., info@)")
  general_contact_webpage: str = Field(description="Contact Us or CSR portal URL")
  date_sources_updated: str = Field(description="Date source last updated/published (YYYY-MM-DD or YYYY)")
  notes: str = Field(description="Brief notes including confidence level")


class CompanyType(str, Enum):
  CORPORATE_SPONSOR = "Corporate Sponsor"
  PRODUCT_DONOR_LARGE = "Product Donor (Large)"
  PRODUCT_DONOR_SMALL = "Product Donor (Small Biz)"
  RESTAURANT_FB = "Restaurant / F&B"
  MEDIA_PR = "Media / PR"


class AAPIRelevance(str, Enum):
  HIGH = "High"
  MEDIUM = "Medium"
  LOW = "Low"


@dataclass
class CompanyLead:
  company_name: str
  website: str
  hq_location: str
  industry: str
  company_type: CompanyType
  aapi_relevance: AAPIRelevance
  aapi_notes: str
  source: str

  # Enrichment fields are added dynamically by the enrichment agent
  date_added: str = field(
      default_factory=lambda: datetime.now().date().isoformat()
  )

  def to_csv_row(self) -> dict:
    return {
        "Company Name": self.company_name,
        "Website": self.website,
        "HQ Location": self.hq_location,
        "Industry / Category": self.industry,
        "Company Type": (
            self.company_type.value
            if isinstance(self.company_type, CompanyType)
            else self.company_type
        ),
        "AAPI Relevance": (
            self.aapi_relevance.value
            if isinstance(self.aapi_relevance, AAPIRelevance)
            else self.aapi_relevance
        ),
        "AAPI Notes": self.aapi_notes,
        "Source": self.source,
        "Date Added": self.date_added,
    }
