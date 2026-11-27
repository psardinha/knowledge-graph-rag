from openai import OpenAI

GRAPH_EXTRACTION_SYSTEM_PROMPT = """
You are a high-precision entity and relationship extraction system for literary and historical texts.

Extract all identifiable entities and explicitly supported relationships that satisfy the schemas and instructions supplied by the user.

Use exclusively the supplied passage as evidence.
Do not use external knowledge, knowledge of the literary work or information from other passages.

Prioritize accurate identification and classification over speculative or excessive extraction.

Distinguish entities from ordinary words, descriptions, attributes, relationships, activities and incidental references.

Perform both entity extraction and relationship extraction.

Identify, classify and normalize all valid entities in the supplied passage.
Extract their attributes and source evidence according to the supplied rules.
Extract all explicitly supported relationships between the identified entities according to the supplied rules.

Canonical person-name normalization is mandatory.

When a person's personal name is identifiable, PERSON.name must contain the personal name only.
Never include a title, rank, honorific or form of address in PERSON.name when the personal name is identifiable.

When the passage identifies a specific, distinguishable individual but provides no personal name, use the shortest clear textual
designation that identifies that individual as PERSON.name.

Do not extract generic, hypothetical or alternative forms of address as unnamed PERSON entities. Do not invent a personal name.

When a title or rank accompanies a personal name:
- Remove the designation from PERSON.name.
- Extract it as a separate TITLE_OR_RANK entity when supported by the passage.
- Preserve the original wording, including the designation, in PERSON.source_quote.
- Use the normalized PERSON.name in every relationship endpoint.

Before returning the result, review EVERY PERSON.name against the original passage.

Verify that no PERSON.name contains a title, rank, honorific or form of address when the personal name is identifiable.

Verify that all relationship endpoints use the normalized canonical entity names.

Follow the supplied structured output schema exactly.
Return only the structured output requested by the user.
"""



GRAPH_EXTRACTION_PROMPT = """
-OBJECTIVE-

Extract all valid entities and all explicitly supported relationships
from the supplied passage.

Follow the permitted entity types, classification rules,
canonical naming rules, attribute requirements, source-evidence
requirements and relationship rules.

Identify entities first, then extract relationships.
During relationship review, add any independently valid entity that was overlooked and validate it before using it as an endpoint.

-ENTITY TYPES-

Use only the entity types permitted by the structured output schema.

PERSON:
An identifiable individual human being.
An individual may be fully named, partially named or unnamed, provided
the passage identifies that individual sufficiently to distinguish them.

FAMILY:
A family, dynasty or lineage connected by ancestry, blood or marriage.
A family is a collective entity, not an individual person.

ORGANIZATION:
An institution, government, political body, religious institution
or organized social association.
Families are FAMILY.
Military formations are MILITARY_UNIT.

LOCATION:
A geographical or physical place, including countries, regions, cities,
towns, villages, estates, buildings and landmarks.

MILITARY_UNIT:
An organized military formation or subdivision, such as an army,
corps, division, regiment or battalion.

EVENT:
An identifiable and meaningful historical, political, diplomatic,
military, social or personal occurrence.
This includes specific military engagements, such as battles.
A battle is classified as EVENT, with event_type="battle" when
the passage supports that classification.

TITLE_OR_RANK:
A formal aristocratic title, nobility designation, military rank,
political office or recognized court position.

CONCEPT:
An independently identifiable abstract idea, philosophical principle,
social value, political ideology or theme with meaningful semantic
significance in the passage.
An abstract-sounding word or expression is not automatically a concept.

-ENTITY EXTRACTION-

Extract entities that are identifiable and meaningfully mentioned
in the passage.

A valid entity does not need to participate in a relationship.

Do not omit an identifiable entity merely because:
- It is mentioned only once.
- Its full name is not provided.
- Its identity is not fully described.
- Some of its attributes are unknown.
- It has no relationships with other entities.

PERSON IDENTIFICATION:

Extract a PERSON when the passage provides a sufficiently specific reference to distinguish an individual.

A generic form of address, such as "Prince", "Emperor", "Count" or "Sir" does not identify a PERSON by itself.
Do not extract a separate PERSON when such a designation is used
to address someone whose individual identity cannot be established from the passage.

Valid references include:
- A full or partial personal name.
- A surname used to identify an individual.
- A familiar name or nickname.
- A name accompanied by a distinguishing description.
- A specific relational designation, such as someone's wife, husband, son or daughter, when it identifies a particular individual.
- An unnamed individual who is otherwise distinguishable in the passage.

Use the person's personal name as the entity name. Apply the CANONICAL PERSON NAME NORMALIZATION rules below.

A title, rank or designation used without an identifiable personal name must never be extracted as a PERSON solely because it refers
to a human being.

For example, in "the prince replied", extract no PERSON named "Prince" unless the passage independently establishes 
the individual's identity.

The fact that a designation establishes gender does not mean that it establishes the identity or canonical name of a PERSON.

HYPOTHETICAL AND GENERIC FORMS OF ADDRESS

Do not extract hypothetical, generic, or alternative forms of address as PERSON entities.

A title or rank used in an invitation, greeting, example, or hypothetical expression does not establish the existence of an
identifiable individual.

For example, "Count (or Prince)" in a generic invitation is not a PERSON entity and must not generate a HOLDS_TITLE_OR_RANK
relationship.

CANONICAL PERSON NAME NORMALIZATION:

When a person's personal name is provided, the PERSON.name field
must contain the personal name only.

Exclude titles, ranks, honorifics and forms of address regardless
of their position relative to the personal name.

When the passage identifies a distinguishable individual but provides
no personal name, use the shortest clear textual designation that
identifies that individual as PERSON.name. Do not invent a personal
name. Retain the designation needed to distinguish the individual,
even if it is a relational or descriptive expression.

Apply the following procedure to EVERY PERSON entity before returning the result:

1. Examine the original passage for the exact wording used to refer to the person.
2. Determine whether the personal name is accompanied by a title, rank, honorific or form of address, 
   whether it appears before or after the personal name.
3. If it is, remove that designation from PERSON.name.
4. Preserve the designation as a separate TITLE_OR_RANK entity when it represents a formal title, rank or position held by the person.
5. Preserve the original wording, including the designation, in PERSON.source_quote.
6. Use the normalized PERSON.name consistently in all relationship endpoints.

Examples:

- "Prince Vasíli Kurágin" → PERSON.name: "Vasíli Kurágin"; TITLE_OR_RANK.name: "Prince".
- "Empress Márya Fëdorovna" → PERSON.name: "Márya Fëdorovna"; TITLE_OR_RANK.name: "Empress".
- "Count Pierre Bezukhov" → PERSON.name: "Pierre Bezukhov"; TITLE_OR_RANK.name: "Count".
- "General Kutúzov" → PERSON.name: "Kutúzov"; TITLE_OR_RANK.name: "General".

FINAL PERSON NAME VALIDATION:

Review EVERY extracted PERSON independently.

For each PERSON:
1. Identify the personal name in the original passage.
2. Identify any title, rank, honorific or form of address accompanying it.
3. Remove that designation from PERSON.name, regardless of whether it appears before or after the personal name.
4. If a personal name is provided, verify that PERSON.name contains only that name, without titles, ranks, honorifics or 
   forms of address. If no personal name is provided, verify that PERSON.name uses the shortest clear textual designation 
   identifying the individual, without inventing a name.
5. Verify that all relationship endpoints use the normalized name.

This validation is mandatory and applies even when the title and personal name appear together as one continuous expression.

For example:
"Prince Vasíli Kurágin" must produce:
PERSON.name = "Vasíli Kurágin"
TITLE_OR_RANK.name = "Prince"

FAMILY IDENTIFICATION:

Extract FAMILY only when the passage refers to a family, dynasty
or lineage as a collective entity.

A surname used to identify an individual does not automatically
identify a FAMILY.

A reference to someone's relatives, household or possessions does not automatically establish a separate FAMILY entity.

A family name appearing in a possessive expression does not establish that the family itself is being discussed.

Do not extract a FAMILY when the passage refers only to property, estates, residences, possessions or other things associated with
that family.

Extract the FAMILY only when the family, its members, lineage, ancestry or collective identity is itself meaningfully mentioned.

For example, "the family estates of the Buonapartes" identifies property, not necessarily the Buonaparte family 
as a collective entity.

OTHER IDENTIFICATION RULES:

Do not extract:
- Pronouns.
- Generic nouns.
- Ordinary objects or possessions.
- Incidental references with no meaningful relevance.
- Descriptive phrases that do not identify an entity.
- Hypothetical or merely possible entities.
- Generic occupational or social categories.

-ENTITY CLASSIFICATION-

Select the entity type according to the entity's meaning and function
in the supplied passage.

A person is an individual, never a family or organization.
A family is a collective entity, never an individual person.
A military formation is MILITARY_UNIT, not ORGANIZATION.
A specific military engagement is EVENT, not a separate entity type.

A person's occupation must not be confused with a formal title, aristocratic designation, military rank or court position.

TITLE_OR_RANK EXTRACTION:

Extract a TITLE_OR_RANK entity whenever the passage explicitly identifies a title, rank, or formal position as 
held or occupied by a person.
Do not extract titles based solely on a word's possible meaning; the passage must establish that the word is being used 
as an actual title, rank, or formal position.

For example:
- "Prince Vasíli smiled at him" identifies "Vasíli" as the canonical name of a PERSON and "Prince" as a TITLE_OR_RANK.

Do not classify a title or court position as a person's occupation.

Classify entities according to their meaning and function in the passage, not according to other possible meanings of 
their names or expressions.

Do not extract a title or rank when it appears only in a hypothetical,
conditional or speculative form of address and the passage does not
establish that the designation is actually held by an identifiable person.

FINAL TITLE EXTRACTION VALIDATION:

Review and CORRECT the name of EVERY PERSON against EVERY reference to that person in the original passage.

This is a mandatory correction step, not merely a check.

Apply corrections directly to the PERSON entity object in the entities list. Do not merely identify the title or create a
separate TITLE_OR_RANK entity while leaving PERSON.name unchanged.

If PERSON.name contains a title, rank or court position together with an identifiable personal name, overwrite PERSON.name with
the personal name alone.

After correcting PERSON.name, update every relationship endpoint that refers to that PERSON to use the corrected canonical name.

Whenever the passage explicitly establishes that a PERSON holds a formal title, rank or court position:
- Extract the designation as a separate TITLE_OR_RANK entity.
- Exclude the designation from PERSON.name.
- Ensure that PERSON.name contains only the person's personal name.
- Extract a HOLDS_TITLE_OR_RANK relationship from that PERSON to the corresponding TITLE_OR_RANK entity.
- Use the normalized PERSON.name as the relationship source_entity.

Do not omit a title merely because it appears as part of the person's full reference rather than in a separate sentence.

Do not leave the title embedded in PERSON.name merely because the title and personal name appear together in the passage.

Perform this validation independently for EVERY extracted PERSON.

-CONCEPT EXTRACTION-

Extract CONCEPT only when an abstract idea, philosophical principle,
social value, political ideology or theme is independently identifiable
and has meaningful semantic significance in the passage.

The concept must function as an actual subject of thought, discussion,
evaluation or reflection in the passage.
A word or expression does not qualify merely because it can have
an abstract, symbolic or philosophical meaning.

Do not extract:
- Ordinary nouns, activities or phenomena.
- Objects, possessions or resources.
- Generic descriptions or descriptive noun phrases.
- Metaphorical or figurative expressions used only rhetorically.
- Ideas that are merely implied or inferred from the passage.
- Words that have an abstract meaning in other contexts but do not function as meaningful concepts in the passage.
- Personified figures, supernatural beings or epithets referring to identifiable entities.
- rhetorical, figurative, religious, or hypothetical references as CONCEPT entities merely because they are expressed as nouns.

Extract a CONCEPT only when it represents an independently meaningful subject, idea, theory, or phenomenon relevant to
the passage.

Do not infer concepts solely from the subject matter, setting or atmosphere of the passage.

Do not classify diseases, disorders, symptoms, or medical conditions as CONCEPT entities.

If the schema does not provide an appropriate entity type for them, do not extract them.


-EVENT EXTRACTION-

Extract EVENT only when the passage presents an identifiable occurrence
as a meaningful event in its own right.

This includes significant military engagements, diplomatic occurrences,
political developments, social occasions and personal events.

A battle is an EVENT. When supported by the passage, identify its
event_type as "battle".

Do not extract:
- Routine activities.
- Ordinary conversations or interactions.
- Incidental references to occasions.
- Generic gatherings or meetings that have no meaningful significance
  in the passage.
- Dates or time expressions without an associated identifiable event.

A reception, meeting or gathering is not automatically an EVENT
merely because it is mentioned.
Extract it only when the passage presents it as a meaningful occurrence
in its own right.

A date is an attribute of an event when the passage explicitly
associates that date with the event.
A date alone is not an event.

-ATTRIBUTE EXTRACTION-

Extract only attributes permitted by the entity's schema.

Populate an attribute only when the supplied passage explicitly supports
that specific attribute.

Do not infer attributes from:
- Historical knowledge.
- A person's name.
- Stereotypes or general assumptions.
- A presumed nationality or geographical origin.
- Knowledge of the novel or its characters.
- Information from other passages.
- Another entity's attributes or identity.

Gender may be established by explicit grammatical references or inherently gender-specific designations, as defined in the GENDER section.
Do not use titles or designations to infer attributes other than gender unless the passage explicitly supports those attributes.

Attributes may be extracted from explicit textual evidence, including unambiguous grammatical references.

Descriptions must be concise, useful and factually grounded.
A description may summarize an explicit statement about an entity, but must not introduce unsupported information or relationships.

Do not repeat the entity's name as its description.

For locations, do not use general geographical knowledge to supply a country, region or place classification that the passage does
not explicitly establish.

If an attribute is not explicitly supported by the passage, return null.

Never use strings such as "unknown", "unspecified", "not stated", "not specified" or "N/A" as attribute values.

An entity may be extracted with only its entity_type and name.
Incomplete attributes are preferable to invented information.

PERSON DESCRIPTION:

For each PERSON, provide a concise description summarizing the
most relevant information explicitly supported by the supplied passage.

The description may include:
- Physical appearance.
- Personality and behavioural characteristics.
- Occupation, social status and position.
- Background and experiences.
- Relevant actions, roles and relationships.

Combine related information into a coherent description rather than listing isolated facts.

Use only information explicitly supported by the supplied passage.
Do not use external knowledge, assumptions or information from other passages.

Do not infer nationality, geographical origin, identity or other attributes from a person's name, title or presumed identity.

Do not simply repeat the person's name or source_quote.

When the passage provides little information, keep the description correspondingly brief.

Return null when the passage provides no meaningful information suitable for a concise description.

GENDER:

Gender determination is mandatory for EVERY extracted PERSON.

Before returning the result, evaluate the gender of each PERSON using the entire original passage.

Follow this procedure for each PERSON:

1. Examine every reference to the person in the passage, including their canonical name, titles, ranks, occupations,
   court positions, descriptions and grammatical references.

2. Check for unambiguous gender-specific evidence.

3. Set gender to "male" or "female" whenever such evidence exists.

4. Return null only when the passage provides no reliable evidence establishing either gender.

Gender evidence includes:

- Explicit grammatical references, including he/him/his and she/her.
- Gender-specific titles, ranks and forms of address.
- Gender-specific occupations and court positions.
- Explicit descriptions identifying someone as a man or woman.

Examples of gender-specific designations include:

Male:
- Prince
- Emperor
- King
- Duke
- Gentleman

Female:
- Princess
- Empress
- Queen
- Duchess
- Maid of honor
- Lady-in-waiting

IMPORTANT:

The designation "maid of honor" explicitly identifies a female person.
Whenever a PERSON is described as a maid of honor, set gender="female".

The designation "lady-in-waiting" explicitly identifies a female person.
Whenever a PERSON is described as a lady-in-waiting, set gender="female".

Gender evidence must be evaluated independently of canonical name normalization and entity classification.

Do not discard gender evidence because a title or designation is stored as a separate TITLE_OR_RANK entity.

Do not infer gender from a person's name, appearance, presumed identity,
or an occupation or position that is not inherently gender-specific.

Do not return null when unambiguous gender-specific evidence exists.

FINAL GENDER VALIDATION:

Before returning the structured result, verify that the gender of every PERSON has been evaluated using all available evidence in
the supplied passage.

Do not return null when unambiguous grammatical or gender-specific evidence establishes the person's gender.

Return null only when the passage does not provide sufficient evidence to determine gender.


PERSON EVIDENCE SENTENCES:

For each already identified PERSON, collect exact original sentences that provide meaningful information about that individual's biography, appearance, personality, actions, speech, thoughts, relationships, background, status or experiences.

Evidence collection must take place AFTER identifying the PERSON and establishing their canonical name.

A sentence may refer to an already identified person by:
- Their canonical name.
- An explicitly established alias.
- A title or designation.
- A pronoun or another unambiguous reference.

In particular:
- A title such as "Prince" or "Count" is not a PERSON name.
- A shortened name is not a new PERSON when the passage establishes that it refers to an already identified individual.
- A signature or alternative name must not create a new PERSON when the passage establishes that it belongs to an already identified individual.

Do not assume that two different names identify the same person without evidence in the supplied passage.

Collect sentences describing or establishing:
- Physical appearance and personality.
- Actions and behaviour.
- Speech and conversations.
- Thoughts, feelings and intentions.
- Occupation, social status, titles and positions.
- Family, social and personal relationships.
- Background, experiences and relevant events.

Preserve each sentence exactly as it appears in the source passage. Do not paraphrase, summarize, combine sentences or invent information.

Exclude:
- Sentences that merely mention the person without providing meaningful information.
- Signatures or name references that provide no additional information about the person's profile.
- Sentences referring to a person whose identity cannot be established.

Evidence collection must never create or identify a PERSON. It applies only to PERSON entities already identified during entity extraction.

Return an empty evidence_sentences list when no meaningful evidence is available.

FINAL PERSON EVIDENCE VALIDATION:

For every extracted PERSON, review the entire passage again.

Check every sentence containing:
- The person's canonical name or an established alias.
- A title or designation referring to that person.
- A pronoun or other unambiguous reference to that person.

Include every such sentence that provides meaningful information about the person's profile, experiences, actions, condition,
relationships or circumstances.

Do not omit relevant sentences merely because they occur separately from the sentence introducing the person.

Preserve the original wording exactly.

-SOURCE QUOTATION RULES-

Every non-null source_quote must be an exact, contiguous substring of the supplied passage.

Never paraphrase, reconstruct or invent a quotation.

Select the shortest quotation that provides sufficient contextual evidence.

-ENTITY SOURCE EVIDENCE-

For every extracted entity, provide a source_quote containing a short, contextualized exact quotation from the supplied passage 
that directly supports the identification of that entity.

The source_quote does not need to contain evidence for every extracted attribute. However, when the passage explicitly supports 
an extracted attribute through a title, rank, occupation, court position or other designation, include that designation in the 
source_quote whenever it appears in the same passage context as the entity.

For PERSON entities, this is particularly important when a gender-specific designation provides evidence for the person's gender.

For example:

Passage:
"Anna Pávlovna Schérer, maid of honor and favorite of the Empress Márya Fëdorovna"

PERSON:
- name: "Anna Pávlovna Schérer"
- gender: "female"
- source_quote: "Anna Pávlovna Schérer, maid of honor"

The source_quote must:
- Provide direct evidence for identifying or distinguishing the entity.
- Include relevant context and explicit designations that support extracted attributes.
- Be as short as possible while retaining sufficient context and evidence.

For an unnamed or partially named PERSON, select a quotation that establishes the specific individual being referred to.
For example, a reference such as "my daughter" should be supported by the wording that identifies whose
daughter she is, when that context is available.

Determine all attributes by examining the entire supplied passage, not only the selected source_quote.

A source_quote that omits evidence for an attribute does not invalidate that attribute when the evidence is explicitly 
available elsewhere in the supplied passage.

Do not use a quotation that merely mentions an entity without supporting its identification.

If no useful supporting quotation can be identified, return null.

-RELATIONSHIP EXTRACTION-

Relationship extraction must not change the classification criteria
or cause valid entities to be omitted.

A relationship may be extracted only when:
- Both endpoints are valid entities belonging to the permitted entity types.
- Both endpoints are included in the entities list.
- The passage explicitly supports the specific relationship type.
- The relationship matches the meaning and endpoint restrictions
  defined below.

Do not create an entity solely to satisfy a relationship.

If relationship review reveals an independently valid entity that
was overlooked during entity extraction, add that entity to the entities list
and apply all relevant entity identification, classification,
attribute and source-evidence rules before extracting the relationship.

If an endpoint cannot be identified as a valid entity from the passage,
do not extract the relationship.

For every relationship, provide:
- relationship_type
- source_entity
- source_entity_type
- target_entity
- target_entity_type
- source_quote
- Any additional fields required by the relationship schema.

Use the canonical entity names assigned during entity extraction.

The endpoint type values must correspond to the entity types assigned
during entity extraction.
Use the exact values required by the relationship schema.

The correspondence between entity types and relationship endpoint
type values is:
- PERSON corresponds to Person.
- FAMILY corresponds to Family.
- ORGANIZATION corresponds to Organization.
- LOCATION corresponds to Location.
- MILITARY_UNIT corresponds to MilitaryUnit.
- EVENT corresponds to Event.
- TITLE_OR_RANK corresponds to TitleOrRank.
- CONCEPT corresponds to Concept.

Never use an endpoint type that is not permitted for the corresponding
relationship.

Do not infer relationships from:
- Historical or literary knowledge.
- Assumed family connections.
- Titles, names or occupations alone.
- Another entity's attributes.
- Events or circumstances that merely suggest a possible relationship.
- Information from other passages.

Do not extract a relationship merely because two entities appear
in the same sentence or passage.

Use only the relationship types defined below.

BELONGS_TO_FAMILY:

Connect a PERSON to a FAMILY when the passage explicitly establishes
that the person belongs to that family through ancestry, blood
or marriage.

Direction: PERSON -> FAMILY.

RELATED_TO:

Connect two PERSON entities when the passage explicitly establishes a personal or family relationship.

Permitted relation values and direction:
- "parent of": directional, from parent to child.
- "son of": directional, from son to parent.
- "daughter of": directional, from daughter to parent.
- "married to": symmetric.
- "sibling of": symmetric.
- "cousin of": symmetric.
- "related to": symmetric.

Record each symmetric relationship only once.

Do not use RELATED_TO for professional, social, political
or organizational associations.

AFFILIATED_WITH:

Connect a PERSON to an ORGANIZATION when the passage explicitly
establishes membership, employment, service or another identifiable
organizational affiliation.

Direction: PERSON -> ORGANIZATION.

Do not infer organizational affiliation merely because a person
attends an event held at an institution.

HOLDS_TITLE_OR_RANK:

Connect a PERSON to a TITLE_OR_RANK when the passage explicitly
establishes that the person holds the title, rank or position.

Direction: PERSON -> TITLE_OR_RANK.

COMMANDS:

Connect a PERSON to a MILITARY_UNIT when the passage explicitly
establishes that the person commands that unit.

Direction: PERSON -> MILITARY_UNIT.

PART_OF:

Connect a subordinate MILITARY_UNIT to the larger MILITARY_UNIT
of which it is a component.

Direction: subordinate MILITARY_UNIT -> larger MILITARY_UNIT.

PARTICIPATED_IN:

Connect a PERSON, ORGANIZATION or MILITARY_UNIT to an EVENT
when the passage explicitly establishes its participation.

Direction: PERSON / ORGANIZATION / MILITARY_UNIT -> EVENT.

Use participation_role to describe the type of participation:
- "combat": participation in combat, including fighting in a battle.
- "attendance": being present at an event as an attendee or guest.
- "organization": organizing or arranging the event.
- "witness": explicitly observing or witnessing an event, without necessarily participating in it.
- "other": another explicitly supported form of participation.

Use "combat" only when the event is a battle and the passage establishes participation in combat.

ALLIED_WITH:

Connect two ORGANIZATION entities or two MILITARY_UNIT entities when the passage explicitly establishes an alliance or an allied
relationship between them.

The relationship is symmetric.
Record it only once.

OPPOSED:

Connect two ORGANIZATION entities or two MILITARY_UNIT entities when the passage explicitly establishes that they are opposing
one another in a military, political or organizational conflict.

The relationship is symmetric.
Record it only once.

LOCATED_IN:

Use this relationship for either:
- A LOCATION contained within another LOCATION.
- An EVENT that explicitly takes place at a LOCATION.

Permitted endpoint combinations:
- LOCATION -> LOCATION.
- EVENT -> LOCATION.

For geographical containment, the source is the smaller or contained
location and the target is the larger or containing location.

For event location, the source is the EVENT and the target is the LOCATION.

Use LOCATED_IN only when the passage establishes the actual location or geographical containment.

Do not use it for proximity alone, such as "near Moscow".

Do not infer geographical containment from external knowledge.

-FINAL RELATIONSHIP VALIDATION-

Before returning the result, perform a dedicated relationship-completeness review.

Do not merely validate relationships already extracted. Actively search for missing relationships.

Perform these checks in order:

1. Review every PERSON against every TITLE_OR_RANK. If the passage establishes that the person holds the title or rank, extract HOLDS_TITLE_OR_RANK.
2. Review every pair of PERSON entities for explicitly established family or personal relationships. Extract the permitted RELATED_TO relationship when supported.
3. Review all remaining entity pairs for relationships permitted by the relationship schema.
4. Verify every extracted relationship independently:

- Both endpoints exist in the entities list.
- Both endpoint names exactly match their canonical entity names.
- Both endpoint types match their actual entity types.
- The endpoint combination is permitted for the relationship type.
- The source_quote supports the relationship.

For PARTICIPATED_IN:
- Source: PERSON, ORGANIZATION or MILITARY_UNIT.
- Target: EVENT.

A LOCATION must never be the target of PARTICIPATED_IN.
A person's presence at a location does not establish participation in an event.

Discard relationships that fail validation.

Do not return an empty relationships list until all three relationship-completeness checks have been performed.

RELATIONSHIP DIRECTION AND EVIDENCE VALIDATION

For every relationship, verify that its direction matches the meaning and permitted endpoint combinations of the relationship type.

In particular, LOCATED_IN permits only:
- LOCATION -> LOCATION, where the source is geographically contained within the target.
- EVENT -> LOCATION, where the event explicitly takes place at the target location.

A LOCATION must never be the source of a LOCATED_IN relationship to a PERSON, FAMILY, ORGANIZATION or other non-LOCATION entity.

Ownership, possession, association or control of a location does not establish a LOCATED_IN relationship.

Every relationship must have source evidence that directly supports the specific relationship, not merely the presence of its two
endpoint entities in the same passage.

-RELATIONSHIP EXTRACTION EXAMPLES-

Example 1: Family membership

Passage:
"Pierre belongs to the Bezukhov family."

Extracted entities:
- PERSON: Pierre
- FAMILY: Bezukhov family

Extract:
{{
  "relationship_type": "BELONGS_TO_FAMILY",
  "source_entity": "Pierre",
  "source_entity_type": "Person",
  "target_entity": "Bezukhov family",
  "target_entity_type": "Family",
  "source_quote": "Pierre belongs to the Bezukhov family."
}}

Example 2: Parent-child relationship

Passage:
"Prince Vasíli was the father of Anatole."

Extracted entities:
- PERSON: Vasíli
- PERSON: Anatole

Extract:
{{
  "relationship_type": "RELATED_TO",
  "relation": "parent of",
  "source_entity": "Vasíli",
  "source_entity_type": "Person",
  "target_entity": "Anatole",
  "target_entity_type": "Person",
  "source_quote": "Prince Vasíli was the father of Anatole."
}}

Example 3: Marriage

Passage:
"Pierre was married to Hélène."

Extracted entities:
- PERSON: Pierre
- PERSON: Hélène

Extract:
{{
  "relationship_type": "RELATED_TO",
  "relation": "married to",
  "source_entity": "Pierre",
  "source_entity_type": "Person",
  "target_entity": "Hélène",
  "target_entity_type": "Person",
  "source_quote": "Pierre was married to Hélène."
}}

Record this relationship only once, not again with the
source and target reversed.

Example 4: Organizational affiliation

Passage:
"Anna was employed by the imperial court."

Extracted entities:
- PERSON: Anna
- ORGANIZATION: imperial court

Extract:
{{
  "relationship_type": "AFFILIATED_WITH",
  "source_entity": "Anna",
  "source_entity_type": "Person",
  "target_entity": "imperial court",
  "target_entity_type": "Organization",
  "source_quote": "Anna was employed by the imperial court."
}}

Do not create an affiliation merely because someone attends
an event held at an institution.

Example 5: Title or rank

Passage:
"Anna was appointed maid of honor to the Empress."

Extracted entities:
- PERSON: Anna
- TITLE_OR_RANK: maid of honor

Extract:
{{
  "relationship_type": "HOLDS_TITLE_OR_RANK",
  "source_entity": "Anna",
  "source_entity_type": "Person",
  "target_entity": "maid of honor",
  "target_entity_type": "TitleOrRank",
  "source_quote": "Anna was appointed maid of honor to the Empress."
}}

Example 6: Command

Passage:
"General Kutúzov commanded the Russian army."

Extracted entities:
- PERSON: Kutúzov
- TITLE_OR_RANK: General
- MILITARY_UNIT: Russian army

Extract:

{{
  "relationship_type": "HOLDS_TITLE_OR_RANK",
  "source_entity": "Kutúzov",
  "source_entity_type": "Person",
  "target_entity": "General",
  "target_entity_type": "TitleOrRank",
  "source_quote": "General Kutúzov"
}}

Extract:

{{
  "relationship_type": "COMMANDS",
  "source_entity": "Kutúzov",
  "source_entity_type": "Person",
  "target_entity": "Russian army",
  "target_entity_type": "MilitaryUnit",
  "source_quote": "General Kutúzov commanded the Russian army."
}}


Example 7: Military unit hierarchy

Passage:
"The regiment formed part of the division."

Extracted entities:
- MILITARY_UNIT: regiment
- MILITARY_UNIT: division

Extract:
{{
  "relationship_type": "PART_OF",
  "source_entity": "regiment",
  "source_entity_type": "MilitaryUnit",
  "target_entity": "division",
  "target_entity_type": "MilitaryUnit",
  "source_quote": "The regiment formed part of the division."
}}

Example 8: Participation in a battle

Passage:
"The Russian army fought in the Battle of Borodino."

Extracted entities:
- MILITARY_UNIT: Russian army
- EVENT: Battle of Borodino

Extract:
{{
  "relationship_type": "PARTICIPATED_IN",
  "participation_role": "combat",
  "source_entity": "Russian army",
  "source_entity_type": "MilitaryUnit",
  "target_entity": "Battle of Borodino",
  "target_entity_type": "Event",
  "source_quote": "The Russian army fought in the Battle of Borodino."
}}

Example 9: Alliance

Passage:
"The French army and the Austrian army formed an alliance."

Extracted entities:
- MILITARY_UNIT: French army
- MILITARY_UNIT: Austrian army

Extract:
{{
  "relationship_type": "ALLIED_WITH",
  "source_entity": "French army",
  "source_entity_type": "MilitaryUnit",
  "target_entity": "Austrian army",
  "target_entity_type": "MilitaryUnit",
  "source_quote": "The French army and the Austrian army formed an alliance."
}}

Record the relationship only once.

Example 10: Opposition

Passage:
"The French army opposed the Russian army."

Extracted entities:
- MILITARY_UNIT: French army
- MILITARY_UNIT: Russian army

Extract:
{{
  "relationship_type": "OPPOSED",
  "source_entity": "French army",
  "source_entity_type": "MilitaryUnit",
  "target_entity": "Russian army",
  "target_entity_type": "MilitaryUnit",
  "source_quote": "The French army opposed the Russian army."
}}

Example 11: Geographic containment

Passage:
"St. Petersburg is a city in Russia."

Extracted entities:
- LOCATION: St. Petersburg
- LOCATION: Russia

Extract:
{{
  "relationship_type": "LOCATED_IN",
  "source_entity": "St. Petersburg",
  "source_entity_type": "Location",
  "target_entity": "Russia",
  "target_entity_type": "Location",
  "source_quote": "St. Petersburg is a city in Russia."
}}

Example 12: Event location

Passage:
"The Battle of Borodino took place at Borodino."

Extracted entities:
- EVENT: Battle of Borodino
- LOCATION: Borodino

Extract:
{{
  "relationship_type": "LOCATED_IN",
  "source_entity": "Battle of Borodino",
  "source_entity_type": "Event",
  "target_entity": "Borodino",
  "target_entity_type": "Location",
  "source_quote": "The Battle of Borodino took place at Borodino."
}}

Example 13: Insufficient evidence

Passage:
"Anna was a favorite of the Empress."

Do not extract RELATED_TO or AFFILIATED_WITH solely because
Anna is described as a favorite.

The relationship is not represented by the permitted relationship types.

Example 14: Co-occurrence without a relationship

Passage:
"Pierre and André attended the reception."

Do not extract RELATED_TO merely because both people attended the same event.

Extract the reception as an EVENT only if it is presented as a meaningful occurrence in its own right. Extract
PARTICIPATED_IN only when the reception qualifies as an EVENT and the person's attendance is established.

Example 15: Implied relationship

Passage:
"Anatole entered the room. Prince Vasíli smiled at him."

Do not infer a family, social or organizational relationship between the two people from this passage.

Example 16: Proximity is not containment or event location

Passage:
"The battle took place near Moscow."

Do not extract LOCATED_IN between the battle and Moscow.
The passage establishes proximity, not that the battle took place
in Moscow.

-CANONICAL NAMES AND ALIASES-

Use one consistent canonical name for each entity.

Use the clearest and most complete personal name explicitly supported by the passage, excluding titles, ranks, 
honorifics and forms of address according to the CANONICAL PERSON NAME NORMALIZATION rules.

Treat different names, nicknames and spelling variations as references
to the same entity only when the passage supports that identification.

Alternative names may be recorded as aliases only when the passage
establishes that they refer to the same entity.

Do not invent aliases.

Do not use titles, ranks, occupations or descriptions as aliases
unless the passage establishes them as alternative references
to the same entity.

Do not merge two entities merely because their names are similar,
their descriptions are similar or they have a possible relationship.

Do not resolve an unidentified person to a historically known person
using external knowledge.

-EXTRACTION COMPLETENESS-

Apply the identification and classification rules consistently to the
entire passage.

Extract every valid entity and every explicitly supported relationship.
Do not omit valid entities because they are mentioned only once or have
incomplete attributes or relationships.

Do not invent entities, attributes or relationships.

-OUTPUT REQUIREMENTS-

Return a single structured result conforming exactly to the ExtractionResult schema provided with the request.

The result must contain:
- entities: a list of extracted entities.
- relationships: a list of extracted relationships.

Each entity must be a flat object containing:
- entity_type
- name
- source_quote
- Any additional attributes permitted by its entity schema.

Each relationship must contain the fields required by its relationship schema.

Do not include fields outside the structured output schema.

Do not use nested attributes or properties objects.

Validate that every PERSON name follows the canonical name normalization rules defined above.
Do not allow titles, ranks, honorifics or forms of address to remain embedded in canonical PERSON names when the personal name is identifiable.

Validate every relationship endpoint against the canonical names and entity types in the entities list.

Return an empty entities list only when no identifiable entities belonging to the permitted entity types are found.

Return an empty relationships list only when no explicitly supported relationships belonging to the permitted relationship types are found.

Do not include explanations, comments, Markdown or code fences.

-ENTITY CLASSIFICATION EXAMPLES-

Example 1: Person with an aristocratic title

**Passage:**

"Prince Vasíli Kurágin was a tall, stout man with a broad face. He was charming and sociable, but also ambitious and calculating.
As a wealthy and influential aristocrat, he frequently attended social gatherings and used his connections to advance his
family's interests."

Extract the PERSON:
- Canonical name: "Vasíli Kurágin".
- Gender: "male", explicitly established by the title "Prince".
- Evidence sentences: all three original sentences, because they provide meaningful information about the person.

Extract the TITLE_OR_RANK:
- Name: "Prince".
- Category: "Aristocratic title".

Extract a HOLDS_TITLE_OR_RANK relationship from PERSON "Vasíli Kurágin" to TITLE_OR_RANK "Prince".

Expected entities:

- Person:
  {{
    "name": "Vasíli Kurágin",
    "entity_type": "PERSON",
    "gender": "male",
    "description": "A wealthy and influential aristocrat, described as tall and stout, with a broad face. He is charming, sociable, ambitious and calculating, and uses his social connections to advance his family's interests.",
    "evidence_sentences": [
      "Prince Vasíli Kurágin was a tall, stout man with a broad face.",
      "He was charming and sociable, but also ambitious and calculating.",
      "As a wealthy and influential aristocrat, he frequently attended social gatherings and used his connections to advance his family's interests."
    ],
    "source_quote": "Prince Vasíli Kurágin"
  }}

- TitleOrRank:
  {{
    "name": "Prince",
    "entity_type": "TITLE_OR_RANK",
    "category": "Aristocratic title",
    "source_quote": "Prince Vasíli Kurágin"
  }}

Additionally, create a HOLDS_TITLE_OR_RANK relationship connecting both entities.

Example 2: Female person with a title

Passage:
"Empress Márya Fëdorovna entered the room."

Extracted entities:

- PERSON:
  {{
    "name": "Márya Fëdorovna",
    "description" : null,
    "entity_type": "PERSON",
    "gender": "female",
    "evidence_sentences": ["Empress Márya Fëdorovna entered the room."],
    "source_quote": "Empress Márya Fëdorovna"
  }}

- TITLE_OR_RANK:
  {{
    "name": "Empress",
    "entity_type": "TITLE_OR_RANK",
    "category": "Aristocratic title",
    "source_quote": "Empress Márya Fëdorovna"
  }}

Extract a HOLDS_TITLE_OR_RANK relationship from "Márya Fëdorovna" to "Empress".

Other examples:

"Anna was a maid of honor" identifies Anna as a PERSON and "maid of honor" as a TITLE_OR_RANK.

"Russian Imperial Army" is a MILITARY_UNIT, not an ORGANIZATION.

"Battle of Borodino" is an EVENT with event_type="battle".

"Family estates" is not a CONCEPT.
It is a descriptive reference to property.

An ordinary mention of "war" is not automatically a CONCEPT.
A possible or implied war must not be extracted as an entity.

A surname identifying one individual is not automatically a FAMILY.

A reference such as "the young prince's wife" may identify a PERSON even when the individual's full name is not supplied.

A generic reference such as "a footman" does not identify a PERSON
unless the passage distinguishes a particular individual.

Do not classify an epithet referring to an individual as a CONCEPT.

Do not infer that a person is Russian merely because the passage
uses a Russian name or aristocratic title.

Do not infer the country or geographical classification of a location
from external geographical knowledge.

-REAL DATA-

Text to extract entities from:

{text}
"""


SYSTEM_CONSOLIDATION_PROMPT = """
You are responsible for resolving duplicate entity records in a knowledge graph extracted from a novel.

Your task is to identify different entity records that refer to the same underlying entity within the supplied chapter.

You will receive:
- A list of entities already consolidated by exact entity name and type.
- Each entity's unique ID, entity type, name, attributes and source evidence.

Use only the information supplied in these entity records.
Do not assume access to the original chapter text or to passages that have not been provided.

ENTITY RESOLUTION RULES:

1. Only merge entities when the supplied evidence supports that they refer to the same underlying entity.

2. Entities must have exactly the same entity type to be merged. Never merge entities of different types.

3. Consider the following when identifying duplicate entities:
   - Full names and partial names.
   - First names and surnames.
   - Familiar names and nicknames.
   - Alternative forms or spellings of a name.
   - Titles and forms of address accompanying personal names.
   - Explicit contextual references establishing that different designations refer to the same individual.

4. A person's title is not part of their personal identity. For example, in the sentence "Prince Vasíli Kurágin", only 
   "Vasíli Kurágin" constitutes the canonical name of the PERSON. The title "Prince" is a separate TITLE_OR_RANK entity and must 
   never be included in the PERSON's canonical name.

5. Do not merge entities merely because their names are similar, share a surname, have the same title, or appear in the same passage.

6. Generic references such as "the prince", "the count", "the officer" or "my aunt" must only be resolved to a named individual when 
   the supplied context clearly establishes that identity.

7. Do not use external knowledge or assumptions about the novel.
   Base every resolution exclusively on the supplied entity records and their source evidence.

8. When the evidence is insufficient or ambiguous, do not merge the entities.

REPRESENTATIVE ENTITY SELECTION:

For each merge group, select one entity as its representative.

1. The representative must be one of the existing entities in the merge group.

2. Select the entity with the most complete and informative canonical name.

3. For PERSON entities, prefer a full personal name over a partial name, first name, surname, familiar name or title-based designation.

4. Prefer a name without titles, ranks or forms of address when the person's personal name is identifiable.

5. If two or more candidates have equally complete and informative names, prefer the entity with the strongest supporting source 
   evidence. If they remain equally suitable, select the entity with the lowest ID.

6. Retain the representative's original entity ID and canonical name. Do not invent a new name or ID.

OUTPUT REQUIREMENTS:

1. Return only merge groups containing at least two distinct entities.

2. Each entity ID must belong to the supplied entity list.

3. Each entity ID must appear exactly once within its merge group. Never repeat an ID.

4. Each entity ID may appear in no more than one merge group across the entire response.

5. Never return duplicate merge groups. Each group of entities referring to the same underlying entity must appear exactly once.

6. The representative_id must identify exactly one entity in the group and must appear exactly once in entity_ids.

7. The entity_ids list must contain all and only the distinct IDs of entities belonging to that same underlying entity.

8. Do not create new entities or entity IDs.

9. Do not include entities that should remain separate.

10. Return an empty merge_groups list if no merges are sufficiently supported.

FINAL VALIDATION BEFORE RESPONDING:

Before producing the response, verify all of the following:
- Every merge group contains at least two distinct entity IDs.
- No entity ID is repeated within a group.
- No entity ID appears in more than one group.
- No merge group is duplicated.
- Every representative_id belongs to its corresponding entity_ids list.
- Every ID exists in the supplied entity list.
- Every group contains only entities of the same entity type.

If any proposed merge group violates these requirements, correct it before returning the response.

Return only the structured response matching the supplied schema.
"""


PERSON_PROFILE_SYSTEM_PROMPT = """\
You are an expert at consolidating information about people extracted from literary and historical texts.

Follow the instructions in the user message carefully.
Use exclusively the supplied information.
Do not introduce external knowledge or unsupported assumptions.
Prioritize factual accuracy and completeness.
Do not omit meaningful information simply to produce a shorter answer.
Treat descriptions, evidence sentences and graph relationships as potentially imperfect extracted data.
Return only the requested output.
"""


PERSON_PROFILE_SUMMARIZE_PROMPT = """
You are an expert at consolidating information about people extracted from literary and historical texts.

Your task is to produce a comprehensive, accurate and coherent profile of the specified person using exclusively the information provided.

The input contains:
- The canonical name of the person.
- A list of descriptions extracted from different text passages.
- A list of evidence sentences associated with the person.
- A list of graph relationships extracted for the person, including relationship types, directions, related entities and 
  relationship attributes.

### Instructions

1. Combine all relevant information into a single, coherent profile.
2. Preserve all meaningful facts, attributes, events, actions, roles, relationships and biographical details contained in the input.
3. Consider all relevant evidence sentences, not just the most prominent or frequently repeated information.
4. Do not discard a meaningful fact merely because it appears in only one evidence sentence.
5. Eliminate redundant information and repeated statements without losing meaningful details.
6. Do not introduce facts, assumptions or external knowledge that are not supported by the supplied information.
7. Do not infer relationships, motivations, intentions or personal characteristics unless explicitly supported by the input.
8. If descriptions and evidence contain contradictory information:
   - Treat evidence sentences as supporting material, not automatically as proof.
   - Evaluate the relevance and reliability of each evidence sentence.
   - Prefer conclusions supported by relevant, explicit evidence over unsupported statements in descriptions.
   - Do not invent explanations or resolve contradictions using external knowledge or assumptions.
   - If the contradiction cannot be resolved using the supplied information, preserve the uncertainty in the summary.
9. Distinguish between facts about the person and opinions, impressions or statements made by other characters.
10. Use third-person narrative and refer to the person by their canonical name or appropriate pronouns.
11. Avoid unnecessary introductory phrases and generic statements.
12. Write a natural, informative and coherent profile.
13. Conciseness must not come at the expense of meaningful information. The objective is to produce a comprehensive profile, 
    not merely a one-sentence identification.

### Evidence evaluation and filtering

Evidence sentences are raw extracted material and must not be assumed to be valid, relevant or reliable.

They may contain:
- Irrelevant or unrelated information.
- Noise and incomplete statements.
- Extraction errors or corrupted text.
- Misleading references or incorrect entity attribution.
- Ambiguous statements that cannot be reliably associated with the specified person.
- Redundant or repetitive information.

You must critically evaluate every evidence sentence before using it.

1. Select only sentences that provide meaningful, relevant and reliable information about the specified person.
2. Discard irrelevant, redundant, corrupted, ambiguous or incorrectly attributed sentences.
3. Do not assume that a sentence refers to the specified person merely because it appears in the supplied evidence list.
4. Distinguish between information that describes the person and information that merely mentions the person in passing.
5. Consider the context of each sentence when determining who is speaking, who is being addressed and to whom a statement refers.
6. Include information about the person's social position, family, relationships, political influence, activities, behaviour and 
   personality whenever supported by the input.
7. Do not use irrelevant evidence to reinforce or justify statements in the descriptions.
8. If the available information is insufficient to establish a fact, do not include it.
9. If relevant sources contradict each other and the contradiction cannot be resolved using the supplied material, preserve the 
   uncertainty rather than inventing an explanation.

### Graph relationship evaluation

The input also contains graph relationships extracted for the specified person from a knowledge graph.

Each relationship record contains:
- relationship: the relationship type.
- direction: OUTGOING or INCOMING, relative to the specified person.
- entity_labels: the labels of the related entity.
- related_entity: the name of the related entity.
- relationship_attributes: the properties associated with the relationship, potentially including source quotes and other contextual information.

Interpret relationship direction as follows:
- OUTGOING: the specified person is the source of the relationship and the related entity is its target.
- INCOMING: the related entity is the source and the specified person is its target.

Evaluate every supplied relationship critically.

1. Consider all relevant relationships when constructing the profile.
2. Use relationship types, direction, related entity information and relationship attributes together to understand the nature 
   of each relationship.
3. Examine source quotes and other evidence attributes whenever available. Use them to assess whether the relationship is supported 
   and to understand its context.
4. Do not assume a relationship is correct merely because it exists in the knowledge graph. It may be incorrectly extracted, 
   incomplete or ambiguous.
5. Do not invent additional relationships or reverse their direction.
6. Do not infer the nature of a relationship beyond what its type, attributes and supporting evidence establish.
7. Distinguish relationships that meaningfully characterize the person from incidental associations.
8. Include meaningful family, personal, social, political, military and organizational relationships when supported.
9. Integrate relevant relationships naturally into the profile. Do not reproduce technical relationship type names or simply 
   list graph records.
10. Avoid repeating information already established by descriptions or evidence sentences.
11. If graph relationships contradict descriptions or evidence sentences, evaluate the available information carefully. 
    Do not automatically prioritize the graph. Preserve unresolved contradictions or uncertainty.
12. If a relationship has no supporting quote or other contextual information, use it cautiously and do not add details beyond 
    what the relationship record explicitly establishes.
13. Identify reciprocal or complementary graph relationships that describe the same underlying relationship. 
    For example, an OUTGOING "parent of" relationship and an INCOMING "son of" relationship between the same two people describe the 
    same parent-child relationship. Consolidate them into a single fact in the profile rather than repeating them.
14. Use the relationship attributes and source quotes from both records when evaluating the relationship. Treat complementary records 
    as additional supporting information, not necessarily as independent evidence.
15. Do not interpret reciprocal records as two separate relationships unless they establish genuinely distinct facts.



### Profile completeness and level of detail

The profile should capture the different dimensions of the person's identity and characterization that can be established from 
the supplied information.

Consider, where supported:
- Identity, titles, social status and position.
- Family members and personal relationships.
- Professional, political, military or social roles.
- Associations and connections with other characters.
- Actions, activities and significant events.
- Behaviour, attitudes and personality traits.
- Influence, reputation and interactions with other people.

These are aspects to consider, not mandatory categories. Do not invent information to fill gaps.

Consolidate related facts into meaningful statements rather than listing individual pieces of evidence or reproducing 
the original sentences.

The length of the profile should be proportional to the amount of relevant information available. A person with substantial 
evidence should receive a sufficiently detailed profile. A person with little relevant evidence should receive a correspondingly 
shorter profile.

### Final validation

Before producing the profile, verify that:

- Every statement is supported by the relevant supplied information.
- No external knowledge or invented details have been introduced.
- Irrelevant or incorrectly attributed evidence has been excluded.
- Meaningful facts from the evidence have not been unnecessarily omitted.
- Redundant information has been consolidated.
- Contradictions have been handled cautiously.
- The profile consistently refers to the specified person.
- The resulting text is coherent, informative and written in the third person.
- Aim to keep the profile within 1,000 characters, including spaces. If necessary, prioritize the most relevant information 
  while preserving meaningful facts. 
  Exceed this limit only when required to avoid omitting important information.

### Insufficient information

If no relevant and reliable information about the person can be identified in the descriptions, evidence sentences or graph relationships, 
return an empty response.
Do not invent a profile merely to produce an answer.


#######
DATA

Person: {person_name}

Descriptions:
{description}

Evidence sentences:
{evidence_list}

Graph relationships:
{relationships}
#######

OUTPUT:
Return only the consolidated profile as plain text.
"""


def get_openai_client():
    #return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
    return OpenAI(base_url=ollama_url, 
                  timeout=1500.0,  # 25 minutes
                  api_key="ollama")


def create_extraction_prompt(input_text):
    return GRAPH_EXTRACTION_PROMPT.format(text = input_text)


def get_summarize_prompt(person_name, description, evidence_list, relationships):
    return PERSON_PROFILE_SUMMARIZE_PROMPT.format(person_name=person_name,
                                                  description=description,
                                                  evidence_list = evidence_list,
                                                  relationships = relationships)