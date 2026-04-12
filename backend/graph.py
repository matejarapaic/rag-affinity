"""
Knowledge graph layer for hybrid RAG retrieval.

Graph structure:
  - Chunk nodes  : id="{doc_id}_{i}"  type="chunk"   (text, doc_id, author_name, chunk_index)
  - Entity nodes : id="ent::{text}"   type="entity"  (entity_type)
  - Edges:
      chunk → entity  : relation="mentions"   (entity appears in chunk)
      chunk ↔ chunk   : relation="adjacent"   (consecutive chunks in same doc)

Retrieval strategy:
  1-hop : chunks that directly mention a query entity     (score += 1.0)
  2-hop : chunks sharing entities with 1-hop chunks       (score += 0.3)

NOTE: The graph is persisted to GRAPH_PATH (default: graph.pkl).
On ephemeral filesystems (e.g. Railway without a volume), the graph resets
on each deploy and must be rebuilt by re-uploading documents.
"""

import logging
import os
import pickle
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

GRAPH_PATH = Path(os.getenv("GRAPH_PATH", "graph.pkl"))

# Lazy-load spaCy so startup is fast even if the model is large
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy  # noqa: PLC0415

            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.info("spaCy model not found — downloading en_core_web_sm…")
                from spacy.cli import download as spacy_download  # noqa: PLC0415

                spacy_download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
        except Exception as exc:
            logger.warning("spaCy unavailable (%s). Graph entity extraction disabled.", exc)
            _nlp = False  # sentinel: tried and failed

    return _nlp if _nlp is not False else None


# ── Graph persistence ────────────────────────────────────────────────────────


def _load_graph() -> nx.Graph:
    if GRAPH_PATH.exists():
        try:
            with open(GRAPH_PATH, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            logger.warning("Could not load graph from disk (%s). Starting fresh.", exc)
    return nx.Graph()


def _save_graph(g: nx.Graph) -> None:
    with open(GRAPH_PATH, "wb") as fh:
        pickle.dump(g, fh)


# Module-level singleton loaded once at startup and shared across requests.
_graph: nx.Graph = _load_graph()


# ── Entity extraction ────────────────────────────────────────────────────────

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "MONEY", "FAC"}


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Return [(normalized_text, entity_type), ...] for named entities in text."""
    nlp = _get_nlp()
    if not nlp:
        return []

    doc = nlp(text[:10_000])  # cap to prevent very slow NLP on huge chunks
    seen: set[str] = set()
    result: list[tuple[str, str]] = []

    for ent in doc.ents:
        if ent.label_ in _ENTITY_LABELS:
            norm = ent.text.strip().lower()
            if norm and len(norm) > 1 and norm not in seen:
                seen.add(norm)
                result.append((norm, ent.label_))

    return result


# ── Graph building (called during ingestion) ─────────────────────────────────


def add_document_to_graph(doc_id: str, chunks: list[str], author_name: str) -> None:
    """
    Add all chunks of a document and their entity relationships to the graph.
    Called once per document at ingest time. Not thread-safe — call sequentially.
    """
    g = _graph
    prev_node: str | None = None

    for i, text in enumerate(chunks):
        chunk_node = f"{doc_id}_{i}"

        g.add_node(
            chunk_node,
            type="chunk",
            text=text,
            doc_id=doc_id,
            author_name=author_name,
            chunk_index=i,
        )

        # Link consecutive chunks within the same document
        if prev_node is not None:
            g.add_edge(prev_node, chunk_node, relation="adjacent")
        prev_node = chunk_node

        # Link chunk → each named entity it mentions
        for entity_text, entity_type in extract_entities(text):
            entity_node = f"ent::{entity_text}"
            if not g.has_node(entity_node):
                g.add_node(entity_node, type="entity", entity_type=entity_type)
            g.add_edge(chunk_node, entity_node, relation="mentions")

    _save_graph(g)
    logger.info(
        "Graph updated | doc=%s  chunks=%d  total_nodes=%d  total_edges=%d",
        doc_id,
        len(chunks),
        g.number_of_nodes(),
        g.number_of_edges(),
    )


# ── Graph retrieval (called during query) ────────────────────────────────────


def graph_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Find chunks related to the query via entity graph traversal.
    Returns matches in the same dict format as Pinecone results so they can
    be merged transparently in hybrid_retrieve().

    Scoring:
      +1.0  per query entity directly mentioning a chunk   (1-hop)
      +0.3  per query entity reachable via a shared entity  (2-hop)
    """
    g = _graph
    if g.number_of_nodes() == 0:
        return []

    query_entities = extract_entities(query)
    if not query_entities:
        # No named entities in query — skip graph search rather than returning noise
        return []

    chunk_scores: dict[str, float] = {}

    for entity_text, _ in query_entities:
        entity_node = f"ent::{entity_text}"
        if not g.has_node(entity_node):
            continue

        # 1-hop: chunks directly mentioning this entity
        one_hop = [
            n for n in g.neighbors(entity_node)
            if g.nodes[n].get("type") == "chunk"
        ]
        for chunk_id in one_hop:
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + 1.0

        # 2-hop: chunks sharing any entity with a 1-hop chunk
        for chunk_id in one_hop:
            for neighbor in g.neighbors(chunk_id):
                if g.nodes[neighbor].get("type") != "entity":
                    continue
                for second_chunk in g.neighbors(neighbor):
                    if (
                        g.nodes[second_chunk].get("type") == "chunk"
                        and second_chunk not in chunk_scores
                    ):
                        chunk_scores[second_chunk] = chunk_scores.get(second_chunk, 0.0) + 0.3

    if not chunk_scores:
        return []

    top = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for chunk_id, score in top:
        node_data = g.nodes[chunk_id]
        results.append(
            {
                "id": chunk_id,
                "score": score,
                "metadata": {
                    "doc_id": node_data.get("doc_id", "unknown"),
                    "author_name": node_data.get("author_name", "unknown"),
                    "text": node_data.get("text", ""),
                },
                "source": "graph",
            }
        )

    return results


# ── Introspection (used by /graph/stats and /graph/debug endpoints) ───────────


def get_stats() -> dict:
    """Return a summary of the current graph for monitoring/debugging."""
    g = _graph
    chunk_nodes = [(n, d) for n, d in g.nodes(data=True) if d.get("type") == "chunk"]
    entity_nodes = [(n, d) for n, d in g.nodes(data=True) if d.get("type") == "entity"]

    top_entities = sorted(
        [
            {"text": n.replace("ent::", ""), "type": d.get("entity_type", "?"), "connections": g.degree(n)}
            for n, d in entity_nodes
        ],
        key=lambda x: x["connections"],
        reverse=True,
    )[:20]

    return {
        "nodes": {
            "total": g.number_of_nodes(),
            "chunks": len(chunk_nodes),
            "entities": len(entity_nodes),
        },
        "edges": g.number_of_edges(),
        "top_entities": top_entities,
    }
