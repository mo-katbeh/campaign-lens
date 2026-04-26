import { ArrowUpRight, GitCompareArrows, Layers3 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatSimilarity, truncateText } from "@/lib/format";
import type { CampaignRecommendation } from "@/lib/types";

interface RecommendationCardProps {
  recommendation: CampaignRecommendation;
}

export function RecommendationCard({ recommendation }: RecommendationCardProps) {
  return (
    <Card className="border-border/80 bg-card/80 shadow-none">
      <CardHeader className="gap-3 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-base">{recommendation.title || "Untitled campaign"}</CardTitle>
            <div className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              <ArrowUpRight className="h-3.5 w-3.5" />
              Campaign {recommendation.campaign_id}
            </div>
          </div>
          <Badge>Similarity {formatSimilarity(recommendation.score)}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 p-5 pt-0">
        <div className="rounded-md border border-border/70 bg-background/40 p-4">
          <p className="text-sm leading-6 text-slate-100">
            {truncateText(recommendation.representative_text, 320) || "No preview text available for this recommendation."}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">
            <GitCompareArrows className="mr-1 h-3.5 w-3.5" />
            Supporting chunks: {recommendation.supporting_chunk_count}
          </Badge>
          {recommendation.representative_chunk_id && (
            <Badge variant="secondary">
              <Layers3 className="mr-1 h-3.5 w-3.5" />
              Representative chunk: {recommendation.representative_chunk_id}
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
