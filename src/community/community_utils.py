from neo4j import Driver

  
def get_person_relationships(driver: Driver, person_name: str, relationships: list[str]):
  records, _, _ = driver.execute_query("""
                                       MATCH (p:Person {name: $person_name})-[r]-(other)
                                       WHERE type(r) IN $relationship_types
                                       RETURN type(r) AS relationship,
                                              CASE WHEN startNode(r) = p THEN 'OUTGOING'
                                                   ELSE 'INCOMING'
                                              END AS direction,
                                              labels(other) AS entity_labels,
                                              other.name AS related_entity,
                                              properties(r) AS relationship_attributes
                                       ORDER BY relationship, related_entity""",
                                       person_name=person_name,
                                       relationship_types = relationships)
  return [record.data() for record in records]


def get_people (driver: Driver, relationships: list[str]):
  candidates_to_summarize, _, _ = driver.execute_query(
        """MATCH (p:Person)
           RETURN p.name AS name, p.description AS description, p.evidence_sentences AS evidence_list""")
  candidates = [{**record.data(), "evidence_list": record["evidence_list"] or []} for record in candidates_to_summarize]
  for candidate in candidates:

    candidate["relationships"] = get_person_relationships(driver, candidate["name"], relationships)
  return candidates


def import_people_summary(driver: Driver, entity_info):
  driver.execute_query("""UNWIND $data AS row
                          MATCH (p:Person {name: row.name})
                          SET p.summary = row.summary""", 
                          data=entity_info)


def calculate_communities(driver: Driver):
    graph_name = "characterGraph"

    # Drop graph if it already exists
    try:
        driver.execute_query("CALL gds.graph.drop($graph_name)", graph_name=graph_name)
    except Exception:
        pass

    # Project Person-to-Person relationships as an undirected graph
    driver.execute_query("""MATCH (source:Person)-[r]->(target:Person)
                            RETURN gds.graph.project($graph_name, source, target, {}, {undirectedRelationshipTypes: ['*']})""",
                         graph_name=graph_name)

    # Calculate Louvain communities and write community ID to Person nodes
    records, _, _ = driver.execute_query("""CALL gds.louvain.write($graph_name, {writeProperty: 'louvain'})
                                            YIELD communityCount, modularity, modularities
                                            RETURN communityCount, modularity, modularities""", 
                                         graph_name=graph_name)
    return [record.data() for record in records][0]



def get_community_info(driver: Driver):
  community_info_query = """MATCH (p:Person)
                            WHERE p.louvain IS NOT NULL

                            WITH p.louvain AS communityId, collect(p) AS nodes
                            WHERE size(nodes) > 1

                            UNWIND nodes AS n
                            MATCH (n)-[r]-(other:Person)
                            WHERE other.louvain = communityId

                            WITH communityId, nodes, collect(DISTINCT r) AS relationships

                            RETURN communityId,
                                   [n IN nodes | {id: n.name, summary: n.summary, labels: labels(n)}] AS nodes,
                                   [r IN relationships | {start: startNode(r).name, 
                                                          type: type(r),
                                                          end: endNode(r).name,
                                                          properties: properties(r)}] AS rels

                            ORDER BY communityId"""
  community_info, _, _ = driver.execute_query(community_info_query)
  return community_info

def add_community_info(driver: Driver, community_info):
  import_community_query = """UNWIND $data AS row
                              MERGE (c:Community {communityId: row.communityId})
                              SET c.title = row.community.title,
                                  c.summary = row.community.summary,
                                  c.rating = row.community.rating,
                                  c.rating_explanation = row.community.rating_explanation
                              WITH c, row
                              UNWIND row.nodes AS node
                              MATCH (n:Person {name: node})
                              MERGE (n)-[:InCommunity]->(c)
                              RETURN count(DISTINCT c) AS communitiesProcessed, count(*) AS membershipsProcessed"""

  records, summary, keys = driver.execute_query(import_community_query, data=community_info)
  return [record.data() for record in records]


def global_searcher(driver: Driver, model_invocation, query: str, rating_threshold: float = 0.0) -> str:
  community_data, _, _ = driver.execute_query("""MATCH (c:Community)
                                                 WHERE c.rating >= $rating
                                                 RETURN c.communityId AS communityId,
                                                        c.title AS title,
                                                        c.summary AS summary,
                                                        c.rating AS rating,
                                                        c.rating_explanation AS ratingExplanation
                                                 ORDER BY c.rating DESC""",
                                              rating=rating_threshold)

  #print(f"global_searcher:: Retrieved {len(community_data)} community reports")
  if not community_data:
     return "global_searcher:: No relevant community reports were found."

  community_data = [{"communityId": community["communityId"],
                     "summary": f"""communityID: {community['communityId']}
                                    title: {community['title']}
                                    rating: {community['rating']}
                                    rating explanation: {community['ratingExplanation']}
                                    summary: {community['summary']}"""} for community in community_data]


  return model_invocation(community_data, query)


def retrieve_people(driver: Driver):
  people, _, _ = driver.execute_query("""MATCH (p:Person)
                                         WHERE p.summary IS NOT NULL  AND trim(p.summary) <> ''
                                         RETURN p.name AS name, p.summary AS summary""")
  return people


def store_people_encoding(driver: Driver, data, encoding_size: int, embeddings_name: str = "person_embeddings"):
  # Store embeddings on Person nodes
  driver.execute_query("""UNWIND $data AS row
                          MATCH (p:Person {name: row.name})
                          CALL db.create.setNodeVectorProperty(p, 'embedding', row.embedding)""", 
                       data=data)

  # Create vector index
  query = f"""CREATE VECTOR INDEX {embeddings_name} IF NOT EXISTS
              FOR (p:Person) ON (p.embedding)
              OPTIONS {{indexConfig: {{`vector.dimensions`: $encoding_size,
                      `vector.similarity_function`: 'cosine'}}}}"""
  driver.execute_query(query, encoding_size=encoding_size)


def local_search(driver: Driver, query_embedding, k_entities, topChunks, topCommunities):
  local_search_query = """CALL db.index.vector.queryNodes('person_embeddings', $k, $embedding)
                          YIELD node, score
                          WITH collect({person: node, score: score}) AS matches
                          WITH matches, [x IN matches | x.person] AS people

                          // Retrieve and rank text chunks
                          CALL {
                              WITH people
                              UNWIND people AS p
                              MATCH (c:Chunk)-[:MENTIONS]->(p)

                              WITH c, count(DISTINCT p) AS freq
                              ORDER BY freq DESC, c.name
                              LIMIT $topChunks

                              RETURN collect({
                                  text: c.content,
                                  relevance: freq
                              }) AS chunks
                          }

                          // Retrieve community reports
                          CALL {
                              WITH people
                              UNWIND people AS p
                              MATCH (p)-[:InCommunity]->(c:Community)

                              RETURN collect(DISTINCT {
                                  communityId: c.communityId,
                                  title: c.title,
                                  summary: c.summary,
                                  rating: c.rating
                              }) AS reports
                          }

                          // Retrieve relationships between matched people
                          CALL {
                              WITH people
                              UNWIND people AS p
                              MATCH (p)-[r]-(m:Person)
                              WHERE m IN people
                                AND elementId(p) < elementId(m)

                              RETURN collect({
                                  source: startNode(r).name,
                                  relationship: type(r),
                                  target: endNode(r).name,
                                  properties: properties(r)
                              }) AS relationships
                          }

                          // Return complete context
                          RETURN {
                              Entities: [
                                  x IN matches | {
                                      name: x.person.name,
                                      description: x.person.description,
                                      evidence: x.person.evidence_sentences,
                                      similarity: x.score
                                  }
                              ],
                              Chunks: chunks,
                              Reports: reports,
                              Relationships: relationships
                          } AS context"""  

  # Retrieve context from Neo4j
  records, _, _ = driver.execute_query(local_search_query, embedding=query_embedding, k=k_entities, topChunks=topChunks)
  return records
