// RyuGraph schema for the Memex entity & citation graph.
// Applied idempotently on connection by `memex.index.graph_store.GraphStore.open`.
// See ADR-0005 (RyuGraph replaces Kuzu) and GUIDELINES.md Part IV "The knowledge graph".

CREATE NODE TABLE IF NOT EXISTS Document (
    doc_id STRING,
    title STRING,
    PRIMARY KEY (doc_id)
);

CREATE NODE TABLE IF NOT EXISTS Entity (
    entity_id STRING,
    name STRING,
    kind STRING,
    PRIMARY KEY (entity_id)
);

CREATE NODE TABLE IF NOT EXISTS Concept (
    concept_id STRING,
    name STRING,
    PRIMARY KEY (concept_id)
);

CREATE NODE TABLE IF NOT EXISTS Citation (
    citation_id STRING,
    surface_text STRING,
    external_id STRING,
    PRIMARY KEY (citation_id)
);

CREATE REL TABLE IF NOT EXISTS MENTIONS (
    FROM Document TO Entity,
    confidence DOUBLE
);

CREATE REL TABLE IF NOT EXISTS CITES (
    FROM Document TO Document,
    surface_text STRING,
    confidence DOUBLE
);

CREATE REL TABLE IF NOT EXISTS DEFINES (
    FROM Document TO Concept
);

CREATE REL TABLE IF NOT EXISTS RELATES_TO (
    FROM Entity TO Entity,
    weight DOUBLE
);
