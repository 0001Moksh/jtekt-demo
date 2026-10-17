import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCitationMap,
  citationPreviewLabel,
  compactSourceGroups,
  normalizeCitationMarkdown,
  safeMarkdownUrl
} from "./citationUtils.js";

const sources = [
  {
    id: "pdf-page-3-chunk-31",
    source_file: "document1.pdf",
    location: "Page 3",
    section: "1-1. Automatic startup procedure",
    source_url: "https://sharepoint.example/document1.pdf",
    content: "Type change instruction followed by steps 7), 8), and 9)."
  },
  {
    id: "pdf-page-8-chunk-52",
    source_file: "document1.pdf",
    location: "Page 8",
    section: "3-2. Startup procedure of manual operation",
    source_url: "https://sharepoint.example/document1.pdf",
    content: "Manual-operation instructions from a different chunk."
  }
];

test("normalizes repeated verbose citations without exposing citation metadata", () => {
  const answer = "Claim [Source 1: document1.pdf; Page 3; URL: https://secret] " +
    "again [Source 1] and another [Source 2].";
  assert.equal(
    normalizeCitationMarkdown(answer),
    "Claim [Source 1](citation:Source%201) again " +
      "[Source 1](citation:Source%201) and another " +
      "[Source 2](citation:Source%202)."
  );
});

test("stable citation IDs map to the exact retrieved chunk on the same PDF", () => {
  const citations = [
    { citation_id: "Source 1", id: sources[0].id, ...sources[0] },
    { citation_id: "Source 2", id: sources[1].id, ...sources[1] }
  ];
  const citationMap = buildCitationMap(citations, sources);

  assert.equal(citationMap.get("Source 1").content, sources[0].content);
  assert.equal(citationMap.get("Source 2").content, sources[1].content);
  assert.equal(citationMap.get("Source 1").id, "pdf-page-3-chunk-31");
  assert.equal(
    citationPreviewLabel(citationMap.get("Source 1")),
    "document1.pdf · Page 3 · Section 1-1. Automatic startup procedure"
  );
  const repeated = normalizeCitationMarkdown("[Source 1] and [Source 1]");
  assert.equal(
    repeated,
    "[Source 1](citation:Source%201) and [Source 1](citation:Source%201)"
  );
  assert.equal(citationMap.get("Source 1").content, sources[0].content);
});

test("source list groups one document while keeping its distinct chunk references", () => {
  const groups = compactSourceGroups([...sources, sources[0]]);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].label, "document1.pdf");
  assert.equal(groups[0].chunks.length, 2);
  assert.deepEqual(groups[0].locations, ["Page 3", "Page 8"]);
  assert.equal(groups[0].sections.length, 2);
});

test("citation markdown URLs are allowed but unsafe navigation URLs are rejected", () => {
  assert.equal(safeMarkdownUrl("citation:Source%201"), "citation:Source%201");
  assert.equal(safeMarkdownUrl("https://example.com"), "https://example.com");
  assert.equal(safeMarkdownUrl("javascript:alert(1)"), "");
});
