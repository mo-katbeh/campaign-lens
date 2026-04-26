import { Compass } from "lucide-react";
import { useState } from "react";

import { EmptyState } from "@/components/dashboard/EmptyState";
import { LoadingState } from "@/components/dashboard/LoadingState";
import { RecommendationCard } from "@/components/dashboard/RecommendationCard";
import { StatusAlert } from "@/components/dashboard/StatusAlert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { getRecommendations } from "@/lib/api";
import { DEFAULT_RECOMMENDATION_TOP_K, DEFAULT_SEED_CHUNK_LIMIT } from "@/lib/constants";
import type { ApiFilters, RecommendationResponse } from "@/lib/types";

interface RecommendationsTabProps {
  filters?: ApiFilters;
}

export function RecommendationsTab({ filters }: RecommendationsTabProps) {
  const [campaignId, setCampaignId] = useState("");
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRecommend() {
    const parsedCampaignId = Number(campaignId);
    if (!Number.isInteger(parsedCampaignId) || parsedCampaignId <= 0) {
      setError("Enter a valid positive campaign ID to find similar campaigns.");
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const response = await getRecommendations({
        campaign_id: parsedCampaignId,
        filters,
        top_k: DEFAULT_RECOMMENDATION_TOP_K,
        seed_chunk_limit: DEFAULT_SEED_CHUNK_LIMIT,
      });
      setResult(response);
    } catch (requestError) {
      setResult(null);
      setError(requestError instanceof Error ? requestError.message : "Recommendation lookup failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card className="border-border/80 bg-card/80 shadow-none">
        <CardHeader className="p-5">
          <CardTitle>Similar Campaign Recommendations</CardTitle>
          <CardDescription>Start from a known campaign and inspect nearby campaigns supported by similar chunks.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 p-5 pt-0">
          <div className="flex flex-col gap-3 md:flex-row">
            <Input
              inputMode="numeric"
              value={campaignId}
              onChange={(event) => setCampaignId(event.target.value)}
              placeholder="Enter campaign ID"
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  handleRecommend();
                }
              }}
            />
            <Button className="md:w-40" onClick={handleRecommend} disabled={isLoading}>
              <Compass className="h-4 w-4" />
              Find Similar
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">Endpoint: /recommendations</Badge>
            <Badge variant="secondary">Seed chunks: {DEFAULT_SEED_CHUNK_LIMIT}</Badge>
            <Badge variant="outline">Similarity + support count</Badge>
          </div>
        </CardContent>
      </Card>

      {error ? <StatusAlert message={error} /> : null}

      {isLoading ? <LoadingState title="Scoring similar campaigns..." /> : null}

      {!isLoading && result && result.recommendations.length > 0 ? (
        <section className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold">Recommendations</h3>
            <p className="text-sm text-muted-foreground">
              {result.recommendations.length} campaigns matched campaign {result.campaign_id}.
            </p>
          </div>
          <div className="space-y-4">
            {result.recommendations.map((recommendation) => (
              <RecommendationCard key={`${recommendation.campaign_id}-${recommendation.representative_chunk_id ?? "seed"}`} recommendation={recommendation} />
            ))}
          </div>
        </section>
      ) : null}

      {!isLoading && result && result.recommendations.length === 0 ? (
        <EmptyState
          title="No recommendations found"
          description="This campaign did not return similar campaigns under the current filters. Try a different campaign ID or relax the filter set."
        />
      ) : null}

      {!isLoading && !result && !error ? (
        <EmptyState
          title="Find similar campaigns"
          description="Recommendations will appear here with similarity scores, supporting chunk counts, and representative evidence text."
        />
      ) : null}
    </div>
  );
}
