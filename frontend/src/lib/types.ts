export interface RetrievedChunk {
  chunk_id: string;
  score: number | null;
  campaign_id: number | null;
  title: string | null;
  theme: string | null;
  beneficiary: string | null;
  year: string | null;
  funding_ratio: number | null;
  quality: number | null;
  text: string;
  metadata: Record<string, unknown>;
}

export interface AnswerResult {
  question: string;
  answer: string;
  retrieved_chunks: RetrievedChunk[];
  source_campaign_ids: number[];
  source_chunk_ids: string[];
}

export interface CampaignRecommendation {
  campaign_id: number;
  title: string | null;
  score: number;
  supporting_chunk_count: number;
  representative_chunk_id: string | null;
  representative_text: string | null;
  source_chunk_ids: string[];
}

export interface RetrieveResponse {
  query: string;
  filters: ApiFilters | null;
  chunks: RetrievedChunk[];
}

export interface RecommendationResponse {
  campaign_id: number;
  filters: ApiFilters | null;
  recommendations: CampaignRecommendation[];
}

export interface NumericRangeFilter {
  gte?: number;
  lte?: number;
}

export interface ApiFilters {
  theme?: string;
  year?: string;
  funding_ratio?: NumericRangeFilter;
  quality?: {
    gte?: number;
  };
}

export interface RetrieveRequest {
  query: string;
  filters?: ApiFilters;
  top_k?: number;
}

export interface AnswerRequest {
  question: string;
  filters?: ApiFilters;
  top_k?: number;
}

export interface RecommendationRequest {
  campaign_id: number;
  filters?: ApiFilters;
  top_k?: number;
  seed_chunk_limit?: number;
}

export interface FilterDraft {
  theme: string;
  year: string;
  fundingRatioMin: string;
  fundingRatioMax: string;
  qualityMin: number;
}
