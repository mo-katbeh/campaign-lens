export function formatSimilarity(score: number | null | undefined): string {
  if (score === null || score === undefined) {
    return "N/A";
  }
  return score.toFixed(3);
}

export function formatFundingRatio(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function formatQuality(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  return `${value.toFixed(0)}/100`;
}

export function truncateText(value: string | null | undefined, maxLength = 220): string {
  const text = (value ?? "").trim();
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength).trimEnd()}...`;
}
