import { DEFAULT_FILTER_DRAFT, QUALITY_MIN } from "@/lib/constants";
import type { ApiFilters, FilterDraft } from "@/lib/types";

function parseOptionalNumber(value: string): number | undefined {
  if (!value.trim()) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function buildApiFilters(draft: FilterDraft): ApiFilters | undefined {
  const filters: ApiFilters = {};

  if (draft.theme !== DEFAULT_FILTER_DRAFT.theme) {
    filters.theme = draft.theme;
  }

  if (draft.year !== DEFAULT_FILTER_DRAFT.year) {
    filters.year = draft.year;
  }

  const fundingMin = parseOptionalNumber(draft.fundingRatioMin);
  const fundingMax = parseOptionalNumber(draft.fundingRatioMax);
  if (fundingMin !== undefined || fundingMax !== undefined) {
    filters.funding_ratio = {
      ...(fundingMin !== undefined ? { gte: fundingMin } : {}),
      ...(fundingMax !== undefined ? { lte: fundingMax } : {}),
    };
  }

  if (draft.qualityMin > QUALITY_MIN) {
    filters.quality = { gte: draft.qualityMin };
  }

  return Object.keys(filters).length > 0 ? filters : undefined;
}

export function countActiveFilters(filters?: ApiFilters): number {
  if (!filters) {
    return 0;
  }

  let count = 0;
  if (filters.theme) {
    count += 1;
  }
  if (filters.year) {
    count += 1;
  }
  if (filters.funding_ratio?.gte !== undefined || filters.funding_ratio?.lte !== undefined) {
    count += 1;
  }
  if (filters.quality?.gte !== undefined && filters.quality.gte > QUALITY_MIN) {
    count += 1;
  }
  return count;
}
