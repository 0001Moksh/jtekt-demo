const CITATION_PATTERN = /\[Source\s+(\d+)(?::[^\]]*)?\]/g;

export function normalizeCitationMarkdown(text) {
  return text.replace(CITATION_PATTERN, (_, number) => {
    const citationId = encodeURIComponent(`Source ${number}`);
    return `[Source ${number}](citation:${citationId})`;
  });
}

export function buildCitationMap(citations = [], sources = []) {
  const sourcesById = new Map(
    sources.filter((source) => source.id).map((source) => [source.id, source])
  );
  const citationMap = new Map();

  citations.forEach((citation) => {
    const source = sourcesById.get(citation.id);
    if (citation.citation_id && (source || citation.content)) {
      citationMap.set(citation.citation_id, { ...source, ...citation });
    }
  });

  if (citationMap.size === 0) {
    sources.forEach((source, index) => {
      citationMap.set(`Source ${index + 1}`, source);
    });
  }

  return citationMap;
}

export function compactSourceGroups(sources = []) {
  const grouped = new Map();
  sources.forEach((source) => {
    const key = source.document_id || source.source_url || source.source_file || source.id || "Source";
    const group = grouped.get(key) || {
      key,
      label: source.source_file || source.id || "Source",
      source_file: source.source_file,
      source_url: source.source_url,
      chunks: new Map(),
    };
    const chunkKey = source.id || `${source.location}:${source.section}`;
    group.chunks.set(chunkKey, source);
    grouped.set(key, group);
  });

  return [...grouped.values()].map((group) => {
    const chunks = [...group.chunks.values()];
    return {
      ...group,
      chunks,
      locations: [...new Set(chunks.map((chunk) => chunk.location).filter(Boolean))],
      sections: [...new Set(chunks.map((chunk) => chunk.section).filter(Boolean))],
    };
  });
}

export function citationPreviewLabel(citation) {
  return [
    citation.source_file,
    citation.location,
    citation.section ? `Section ${citation.section}` : "",
  ].filter(Boolean).join(" · ");
}

export function safeMarkdownUrl(url) {
  if (url.startsWith("citation:")) return url;
  const protocol = url.trim().match(/^([a-z][a-z\d+.-]*:)/i)?.[1]?.toLowerCase();
  return !protocol || ["http:", "https:", "irc:", "ircs:", "mailto:", "xmpp:"].includes(protocol)
    ? url
    : "";
}
