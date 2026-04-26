import type {
  AnswerRequest,
  AnswerResult,
  RecommendationRequest,
  RecommendationResponse,
  RetrieveRequest,
  RetrieveResponse,
} from "@/lib/types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function postJson<TResponse, TRequest>(path: string, payload: TRequest): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const errorPayload = (await response.json()) as { detail?: string; message?: string };
      message = errorPayload.detail || errorPayload.message || message;
    } catch {
      // Ignore JSON parsing failures and keep the generic message.
    }
    throw new ApiError(message);
  }

  return (await response.json()) as TResponse;
}

export function retrieveChunks(payload: RetrieveRequest) {
  return postJson<RetrieveResponse, RetrieveRequest>("/retrieve", payload);
}

export function getAnswer(payload: AnswerRequest) {
  return postJson<AnswerResult, AnswerRequest>("/answer", payload);
}

export function getRecommendations(payload: RecommendationRequest) {
  return postJson<RecommendationResponse, RecommendationRequest>("/recommendations", payload);
}

export { ApiError };
