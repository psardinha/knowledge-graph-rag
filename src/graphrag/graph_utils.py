from neo4j import GraphDatabase, Driver, ResultSummary
from collections import defaultdict
import os
from dataclasses import dataclass

@dataclass
class InsertSummary:
  nodes_created: int = 0
  nodes_deleted: int = 0
  relationships_created: int = 0
  relationships_deleted: int = 0
  properties_set: int = 0
  labels_added: int = 0

  def add(self, summary: ResultSummary):
    counters = summary.counters
    self.nodes_created += counters.nodes_created
    self.nodes_deleted += counters.nodes_deleted
    self.relationships_created += counters.relationships_created
    self.relationships_deleted += counters.relationships_deleted
    self.properties_set += counters.properties_set
    self.labels_added += counters.labels_added


def get_neo4j_driver(url : str = None,
                     username : str = None,
                     password : str = None) -> Driver:
  url = url or os.environ.get("NEO4J_URI")
  username = username or os.environ.get("NEO4J_USER")
  password = password or os.environ.get("NEO4J_PASSWORD")
  return GraphDatabase.driver(url, auth=(username, password), notifications_min_severity="OFF")


def init_neo4j_constraints(neo4j_driver: Driver, entity_types: list[str]):
  with neo4j_driver.session() as session:
    for entity in entity_types:
      query = f"CREATE CONSTRAINT {entity.lower()}_unique IF NOT EXISTS FOR (e:{entity}) REQUIRE e.name IS UNIQUE"
      session.run(query)


def insert_chunks (driver: Driver, book: str, chapter: str, chunks: list[str], first_chunk_id: int):
  query = f"""
           MERGE (b:Book {{name: $book}})
           MERGE (ch:Chapter {{name: $chapter}})
           MERGE (b)-[:HAS_CHAPTER]->(ch)
           UNWIND range(0, size($chunks) - 1) AS ndx
           MERGE (chk:Chunk {{id: $first_chunk_id + ndx, content: $chunks[ndx]}})
           MERGE (ch)-[:HAS_CHUNK]->(chk)
           """
  with driver.session() as session:
    session.run(query, book=book, chapter=chapter, chunks=chunks, first_chunk_id=first_chunk_id)
    
    
def insert_entities(tx, entities, entities_map) -> InsertSummary:
  total = InsertSummary()
  for item in entities:
    properties = item.get("properties", {}).copy()
    # Automatically detect list-valued properties    
    list_properties = {key: properties.pop(key) for key, value in list(properties.items()) if isinstance(value, list)}
    scalar_updates = "\n".join( f"SET e.`{key}` = coalesce(e.`{key}`, $properties.{key})" for key in properties)
    list_updates = "\n".join(f"""SET e.`{key}` = reduce(acc = coalesce(e.`{key}`, []),
                                                        x IN $list_properties.{key} |
                                                        CASE WHEN x IN acc THEN acc ELSE acc + x END)"""
                             for key in list_properties)
    entity_name = entities_map.get(item["entity_type"], item["entity_type"])

    # Scalar attributes are updated; list ones are appended
    query = f"""MERGE (e:{entity_name} {{name: $name}}) 
                {scalar_updates}
                {list_updates}
                WITH e
                UNWIND $chunks AS chk
                MATCH (c:Chunk {{id: chk}})
                MERGE (c)-[:MENTIONS]->(e)"""
    summary = tx.run(query, name=item["name"], properties=properties, list_properties=list_properties, chunks=item["chunks"]).consume()
    total.add(summary)
  return total

  
def insert_relationships(tx, relationships, relationship_map) -> InsertSummary:
  total = InsertSummary()
  for item in relationships:
    rel_type = relationship_map.get(item["relationship_type"], item["relationship_type"])
    source_type = item["source_entity_type"]
    target_type = item["target_entity_type"]

    # Only scalar attributes are addressed
    properties = item.get("properties", {})
    property_updates = "\n".join(f"SET r.`{key}` = coalesce(r.`{key}`, $properties.{key})" for key in properties)
    query = f"""
             MATCH (source:{source_type} {{name: $source_entity}})
             MATCH (target:{target_type} {{name: $target_entity}})
             MERGE (source)-[r:{rel_type}]->(target)
             {property_updates}
             """
    summary = tx.run(query, source_entity=item["source_entity"], target_entity=item["target_entity"], properties=properties).consume()
    total.add(summary)
  return total


def consolidate_chapter(chapter_results):
  entities_by_key = {}
  relationships_by_key = {}
  entity_id = 1
  for chunk_id, result in chapter_results.items():
    # result contains the entities and relationships extracted per chunk

    # Process entities
    for entity in result.entities:
      key = (entity.entity_type, entity.name.casefold().strip())
      entity_data = entity.model_dump(exclude_none=True)

      if key not in entities_by_key:
        entities_by_key[key] = {"_id_": entity_id, 
                                "entity": entity, 
                                "source_evidence": [{"chunk": chunk_id, "quote": entity.source_quote}] if entity.source_quote else [],
                                "conflicts": {}}
        entity_id+=1
        continue

      stored = entities_by_key[key]
      existing_data = stored["entity"].model_dump()

      for attribute, value in entity_data.items():
        if attribute in ("name", "entity_type"):
          continue
        if attribute == "source_quote":
          if value:
            evidence = {"chunk": chunk_id, "quote": value}
            if evidence not in stored["source_evidence"]:
                stored["source_evidence"].append(evidence)
          continue
        if value is None:
          continue
        existing_value = existing_data.get(attribute)

        if existing_value is None:
          existing_data[attribute] = value
        elif isinstance(existing_value, list) and isinstance(value, list):
          existing_data[attribute] = list(dict.fromkeys(existing_value + value))
        elif existing_value != value:
          stored["conflicts"].setdefault(attribute, []).append({"value": value, "chunk": chunk_id})

      stored["entity"] = type(entity).model_validate(existing_data)

    # Process relationships
    for relationship in result.relationships:
      relationship_data = relationship.model_dump(exclude_none=True)
      relationship_details = tuple(sorted((field, str(relationship_data[field]).casefold().strip())
                                           for field in ("relation", "participation_role") if field in relationship_data))

      key = (relationship.relationship_type,
             relationship.source_entity_type.casefold().strip(),
             relationship.source_entity.casefold().strip(),
             relationship.target_entity_type.casefold().strip(),
             relationship.target_entity.casefold().strip(),
             relationship_details)
      
      if key not in relationships_by_key:
        relationships_by_key[key] = {"relationship": relationship, 
                                     "source_evidence": [{"chunk": chunk_id, "quote": relationship.source_quote}] 
                                                         if relationship.source_quote else [],
                                     "conflicts": {}}
        continue

      stored = relationships_by_key[key]
      existing_data = stored["relationship"].model_dump()

      for attribute, value in relationship_data.items():
        if attribute in ("relationship_type", 
                         "source_entity_type", "source_entity", 
                         "target_entity_type", "target_entity"):
          continue
        if value is None:
          continue
        if attribute == "source_quote":
          evidence = {"chunk": chunk_id, "quote": value}
          if evidence not in stored["source_evidence"]:
            stored["source_evidence"].append(evidence)
          continue
        existing_value = existing_data.get(attribute)
        if existing_value is None:
          existing_data[attribute] = value

        elif isinstance(existing_value, list) and isinstance(value, list):
          existing_data[attribute] = list(dict.fromkeys(existing_value + value))
        elif existing_value != value:
          stored["conflicts"].setdefault(attribute, []).append({"value": value, "chunk": chunk_id})

      stored["relationship"] = type(relationship).model_validate(existing_data)

  return entities_by_key, relationships_by_key


def prepare_entities(entities_by_key):
  entities = []
  for item in entities_by_key.values():
    entity = item["entity"]
    properties = entity.model_dump(exclude_none=True)
    properties.pop("entity_type", None)
    properties.pop("source_quote", None)
    entities.append({"entity_type": entity.entity_type,
                     "name": entity.name,
                     "properties": properties,
                     "chunks": [ev["chunk"] for ev in item["source_evidence"]],
                     "quotes": [ev["quote"] for ev in item["source_evidence"]],
                     "conflicts": item["conflicts"]})
  return entities


def prepare_relationships(relationships_by_key):
  relationships = []
  for item in relationships_by_key.values():
    relationship = item["relationship"]
    properties = relationship.model_dump(exclude_none=True)
    properties.pop("relationship_type", None)
    properties.pop("source_entity", None)
    properties.pop("source_entity_type", None)
    properties.pop("target_entity", None)
    properties.pop("target_entity_type", None)
    relationships.append({"relationship_type": relationship.relationship_type,
                          "source_entity": relationship.source_entity,
                          "source_entity_type": relationship.source_entity_type,
                          "target_entity": relationship.target_entity,
                          "target_entity_type": relationship.target_entity_type,
                          "properties": properties,
                          "source_evidence": item["source_evidence"],
                          "conflicts": item["conflicts"]})
  return relationships


def save_chapter(driver: Driver, 
                 entity_consolidator,
                 chapter_results, 
                 entity_types_map, 
                 relationship_types_map) -> tuple[InsertSummary, InsertSummary]:
  entities_by_key, relationships_by_key = consolidate_chapter(chapter_results)

  print(f"entities_by_key/1: {entities_by_key}")
  merge_rules = entity_consolidator(entities_by_key)
  print(f"Merge_rules: {merge_rules}")
  for rule in merge_rules:
    ref_entity = next((value for value in entities_by_key.values() if value.get("_id_") == rule["representative_id"]), None)
    entities_id_to_remove = set(rule["entity_ids"])
    entities_id_to_remove.discard(rule["representative_id"])
    entities_name_to_remove = set()
    for entity_id_to_remove in entities_id_to_remove:
      entity_to_remove = next((value for value in entities_by_key.values() if value.get("_id_") == entity_id_to_remove), None)
      entities_name_to_remove.add(getattr(entity_to_remove["entity"], "name"))
      for from_attr, to_attr in zip(["evidence_sentences", "name"], ["evidence_sentences", "aliases"]):
        if to_attr in ref_entity["entity"].__class__.model_fields:
          if getattr(ref_entity["entity"], to_attr) == None:
            setattr(ref_entity["entity"], to_attr, [getattr(entity_to_remove["entity"], from_attr)])
          else:
            getattr(ref_entity["entity"], to_attr).append(getattr(entity_to_remove["entity"], from_attr))
      # Update "source_evidence" field
      ref_entity["source_evidence"].extend(item for item in entity_to_remove["source_evidence"] 
                                             if item not in ref_entity["source_evidence"])
    # Remove replicated entities
    entities_by_key = {key: value for key, value in entities_by_key.items() if value.get("_id_") not in entities_id_to_remove}

    # Correct references in the relationships
    for _, value in relationships_by_key.items():
      if (getattr(value["relationship"], "source_entity_type") == getattr(ref_entity["entity"], "entity_type") and 
          getattr(value["relationship"], "source_entity") in entities_name_to_remove):
        setattr(value["relationship"], "source_entity", getattr(ref_entity["entity"], "name"))
      if (getattr(value["relationship"], "target_entity_type") == getattr(ref_entity["entity"], "entity_type") and 
          getattr(value["relationship"], "target_entity") in entities_name_to_remove):
        setattr(value["relationship"], "target_entity", getattr(ref_entity["entity"], "name"))

  print(f"entities_by_key/2: {entities_by_key}")
  print(f"relationships_by_key: {relationships_by_key}")
  entities = prepare_entities(entities_by_key)
  relationships = prepare_relationships(relationships_by_key)

  def save_transaction(tx):
    return (insert_entities(tx, entities, entity_types_map), 
            insert_relationships(tx, relationships, relationship_types_map))

  with driver.session() as session:
    return session.execute_write(save_transaction)
