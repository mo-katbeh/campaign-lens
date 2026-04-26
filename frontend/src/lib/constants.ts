import type { FilterDraft } from "@/lib/types";

export const THEMES = [
  "community",
  "education",
  "emergency",
  "food",
  "housing",
  "medical",
  "orphan_support",
  "other",
  "water",
  "winter",
] as const;

export const YEARS = [
  "all",
  "2015",
  "2016",
  "2017",
  "2018",
  "2019",
  "2020",
  "2021",
  "2022",
  "2023",
  "2024",
  "2025",
  "2026",
  "unknown",
] as const;

export const QUALITY_MIN = 35;
export const QUALITY_MAX = 100;
export const DEFAULT_TOP_K = 6;
export const DEFAULT_RECOMMENDATION_TOP_K = 6;
export const DEFAULT_SEED_CHUNK_LIMIT = 3;

export const DEFAULT_FILTER_DRAFT: FilterDraft = {
  theme: "all",
  year: "all",
  fundingRatioMin: "",
  fundingRatioMax: "",
  qualityMin: QUALITY_MIN,
};
