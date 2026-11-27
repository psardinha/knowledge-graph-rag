from typing import List, Dict, Optional, Literal, Annotated, Union,  get_args
from pydantic import BaseModel, Field, ConfigDict, model_validator
import json


class BaseEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(..., description="Canonical name or designation of the entity.")
    description: Optional[str] = Field(None, description=("Optional concise description of the entity. "
                                                          "Include only information explicitly supported by the passage. "
                                                          "Do not repeat the entity's name or infer additional facts. "
                                                          "Return null when no useful description is supported."))
    source_quote: Optional[str] = Field(None, description=("A short, exact quotation from the source chunk that supports " 
                                                           "the identification of this entity."))

    @model_validator(mode="before")
    @classmethod
    def normalize_unknown_values(cls, data):
      if isinstance(data, dict):
        placeholders = {"unknown", "unspecified", "not stated", "not specified", "n/a", "none", "null"}
        data = data.copy()
        for key, value in data.items():
          if key not in {"name", "entity_type"} and isinstance(value, str) and value.strip().casefold() in placeholders:
            data[key] = None
      return data
    
class Person(BaseEntity):
    """An identifiable individual human being, including unnamed or partially named individuals."""

    entity_type: Literal["PERSON"]
    aliases: Optional[list[str]] = Field(None, description="Alternative names explicitly used for this person.")
    gender: Optional[str] = Field(None, description="Gender only when explicitly established in the passage.")
    occupation: Optional[str] = Field(None, description="A profession or occupation, not an aristocratic title, rank, or court position.")
    nationality: Optional[str] = Field(None, description="Nationality only when explicitly established in the passage.")
    evidence_sentences: list[str] = Field(default_factory=list,
                                          description=("Exact sentences from the source passage providing meaningful "
                                                       "information about this person, including actions, speech, thoughts, "
                                                       "feelings, relationships, appearance, personality, background, "
                                                       "status and experiences. Do not paraphrase or invent sentences."))


class Family(BaseEntity):
    """
    A family, dynasty or lineage connected by ancestry, blood or marriage.
    """

    entity_type: Literal["FAMILY"]
    aliases: Optional[list[str]] = Field( None, description="Alternative family names explicitly used in the passage.")
    social_status: Optional[str] = Field(None, description="Social status only when explicitly stated in the passage.")


class Organization(BaseEntity):
    """
    An institution, government, political body, religious institution or organized social association.

    Families are FAMILY.
    Military formations are MILITARY_UNIT.
    """

    entity_type: Literal["ORGANIZATION"]
    aliases: Optional[list[str]] = Field(None, description="Alternative organization names explicitly used in the passage.")
    organization_type: Optional[str] = Field(None, description="The type of organization, when explicitly established.")
    headquarters: Optional[str] = Field(None, description="Headquarters location, only when explicitly stated.")
    country: Optional[str] = Field(None, description="Country associated with the organization, only when explicitly stated.")


class Location(BaseEntity):
    """
    A geographical or physical place, including countries, regions, cities, towns, villages, estates, buildings and landmarks.
    """

    entity_type: Literal["LOCATION"]
    region: Optional[str] = Field(None, description="Region associated with the location, only when explicitly stated.")
    country: Optional[str] = Field(None, description="Country associated with the location, only when explicitly stated.")


class MilitaryUnit(BaseEntity):
    """
    An organized military formation or subdivision, such as an army, corps, division, regiment or battalion.
    """

    entity_type: Literal["MILITARY_UNIT"]
    aliases: Optional[list[str]] = Field(None, description="Alternative military unit names explicitly used in the passage.")
    unit_type: Optional[str] = Field(None, description="Military formation type, such as army, corps, division or regiment.")
    country: Optional[str] = Field(None, description="Country associated with the military unit, only when explicitly stated.")


class Event(BaseEntity):
    """
    An identifiable historical, political, diplomatic, military, social or personal occurrence. Includes specific military
    engagements such as battles.
    """
    entity_type: Literal["EVENT"]
    event_type: Optional[str] = Field(None, description=("Type of event, such as battle, diplomatic event, "
                                                         "political event, social gathering or personal event. "
                                                         "Only when supported by the passage."))
    date: Optional[str] = Field(None, description="Date or time period explicitly associated with the event.")
    outcome: Optional[str] = Field(None, description="Outcome explicitly stated in the passage.")
    casualties: Optional[str] = Field(None, description="Casualties explicitly reported in the passage.")


class TitleOrRank(BaseEntity):
    """
    A formal aristocratic title, nobility designation, military rank, political office or recognized court position.
    """

    entity_type: Literal["TITLE_OR_RANK"]
    category: Optional[str] = Field(None, description="Classification such as aristocratic title, military rank, political office or court position.")
    hierarchy: Optional[str] = Field(None, description="Position within a hierarchy, only when explicitly established.")
    country: Optional[str] = Field(None, description="Country associated with the title or position, only when explicitly stated.")


class Concept(BaseEntity):
    """
    An independently identifiable abstract idea, philosophical principle, social value, political ideology or recurring theme.
    Ordinary nouns, objects, possessions and incidental expressions are not concepts.
    """

    entity_type: Literal["CONCEPT"]
    category: Optional[str] = Field(None,
        description="Concept classification, such as philosophical principle, social value or political ideology, when supported by the passage.")

ENTITY_MODELS = {
    "PERSON": Person,
    "FAMILY": Family,
    "ORGANIZATION": Organization,
    "LOCATION": Location,
    "MILITARY_UNIT": MilitaryUnit,
    "EVENT": Event,
    "TITLE_OR_RANK": TitleOrRank,
    "CONCEPT": Concept,
}

EntityRecord = Annotated[Union[Person, Family, Organization, Location, MilitaryUnit, Event, TitleOrRank, Concept],
                         Field(discriminator="entity_type")]



entity_union = get_args(EntityRecord)[0]  # unwrap Annotated
entity_classes = get_args(entity_union)
entity_types_map = {get_args(entity_class.model_fields["entity_type"].annotation)[0]: entity_class.__name__
                    for entity_class in entity_classes}


class BaseRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_entity: str = Field(..., description="Canonical name of the source entity.")
    target_entity: str = Field(..., description="Canonical name of the target entity.")
    source_entity_type: str = Field(..., description="Source entity.")
    target_entity_type: str = Field(..., description="Target entity.")
    source_quote: str = Field(..., description="Exact source passage supporting the relationship.")


class BelongsToFamily(BaseRelationship):
    """Person → Family"""
    relationship_type: Literal["BELONGS_TO_FAMILY"]
    source_entity_type: Literal["Person"]
    target_entity_type: Literal["Family"]


class RelatedTo(BaseRelationship):
    """
    Person → Person

    Specific kinship relationship.
    """
    relationship_type: Literal["RELATED_TO"]
    source_entity_type: Literal["Person"]
    target_entity_type: Literal["Person"]
    relation: Literal["parent of", "married to", "sibling of", "cousin of", "related to", "son of", "daughter of"]


class AffiliatedWith(BaseRelationship):
    """
    Person → Organization

    Membership, employment, institutional affiliation
    or other formal association.
    """
    relationship_type: Literal["AFFILIATED_WITH"]
    source_entity_type: Literal["Person"]
    target_entity_type: Literal["Organization"]


class HoldsTitleOrRank(BaseRelationship):
    """Person → TitleOrRank"""
    relationship_type: Literal["HOLDS_TITLE_OR_RANK"]
    source_entity_type: Literal["Person"]
    target_entity_type: Literal["TitleOrRank"]


class Commands(BaseRelationship):
    """Person → MilitaryUnit"""
    relationship_type: Literal["COMMANDS"]
    source_entity_type: Literal["Person"]
    target_entity_type: Literal["MilitaryUnit"]


class PartOf(BaseRelationship):
    """MilitaryUnit → MilitaryUnit"""
    relationship_type: Literal["PART_OF"]
    source_entity_type: Literal["MilitaryUnit"]
    target_entity_type: Literal["MilitaryUnit"]

class ParticipatedIn(BaseRelationship):
    """
    Person / Organization / MilitaryUnit → Event

    Includes combat participation.
    """
    relationship_type: Literal["PARTICIPATED_IN"]
    source_entity_type: Literal["Person", "Organization", "MilitaryUnit"]
    target_entity_type: Literal["Event"]    
    participation_role: Literal["combat", "attendance", "organization", "witness", "other"]


class AlliedWith(BaseRelationship):
    """
    Organization / MilitaryUnit → Organization / MilitaryUnit
    """
    relationship_type: Literal["ALLIED_WITH"]
    source_entity_type: Literal["Organization", "MilitaryUnit"]
    target_entity_type: Literal["Organization", "MilitaryUnit"]  


class Opposed(BaseRelationship):
    """
    Organization / MilitaryUnit → Organization / MilitaryUnit
    """
    relationship_type: Literal["OPPOSED"]
    source_entity_type: Literal["Organization", "MilitaryUnit"]
    target_entity_type: Literal["Organization", "MilitaryUnit"]  


class LocatedIn(BaseRelationship):
    """
    Location / Event → Location
    """
    relationship_type: Literal["LOCATED_IN"]
    source_entity_type: Literal["Event", "Location"]
    target_entity_type: Literal["Location"]      


RelationshipRecord = Annotated[
    Union[BelongsToFamily, RelatedTo, AffiliatedWith, HoldsTitleOrRank, Commands, PartOf, ParticipatedIn, AlliedWith, Opposed, LocatedIn],
    Field(discriminator="relationship_type")]


relationship_union = get_args(RelationshipRecord)[0]  # unwrap Annotated
relationship_classes = get_args(relationship_union)

relationship_types_map = {relationship_class.model_fields["relationship_type"].default: relationship_class.__name__
                          for relationship_class in relationship_classes}
relationship_types_map = {get_args(relationship_class.model_fields["relationship_type"].annotation)[0]: relationship_class.__name__
                          for relationship_class in relationship_classes}


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    entities: list[EntityRecord] = Field(..., description="All entities extracted from the input text.")
    relationships: list[RelationshipRecord] = Field(..., description="All explicitly supported relationships between extracted entities.")


def get_schema_entities():
  entity_classes = BaseEntity.__subclasses__()
  return [cls.__name__ for cls in entity_classes]


class EntityMergeGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    representative_id: int = Field(...,description=("The ID of the entity that will represent the entire merge group. "
                                                    "It must be one of the IDs included in entity_ids. "
                                                    "Select the entity with the most complete and informative canonical name. "
                                                    "Its ID and canonical name will be retained after merging."))
    entity_ids: list[int] = Field(description="IDs of entities that refer to the same real-world entity.")
    #reason: str = Field(description="Brief explanation based on the supplied passage evidence.")


class EntityResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    merge_groups: list[EntityMergeGroup] = Field(default_factory=list,
                                                 description="Groups of two or more entity IDs referring to the same entity.")
