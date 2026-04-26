import { Database, FileText, Layers3 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatFundingRatio, formatQuality, formatSimilarity, truncateText } from "@/lib/format";
import type { RetrievedChunk } from "@/lib/types";

interface ChunkCardProps {
  chunk: RetrievedChunk;
}

export function ChunkCard({ chunk }: ChunkCardProps) {
  return (
    <Card className="border-border/80 bg-card/80 shadow-none">
      <CardHeader className="gap-3 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-base">{chunk.title || "Untitled campaign"}</CardTitle>
            <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
              {chunk.campaign_id !== null && (
                <span className="inline-flex items-center gap-1">
                  <Database className="h-3.5 w-3.5" />
                  Campaign {chunk.campaign_id}
                </span>
              )}
              <span className="inline-flex items-center gap-1">
                <Layers3 className="h-3.5 w-3.5" />
                Chunk {chunk.chunk_id}
              </span>
            </div>
          </div>
          <Badge variant="default">Score {formatSimilarity(chunk.score)}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 p-5 pt-0">
        <div className="rounded-md border border-border/70 bg-background/40 p-4">
          <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-muted-foreground">
            <FileText className="h-3.5 w-3.5" />
            Retrieved snippet
          </div>
          <p className="text-sm leading-6 text-slate-100">{truncateText(chunk.text, 320) || "No snippet available."}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">Theme: {chunk.theme || "N/A"}</Badge>
          <Badge variant="secondary">Year: {chunk.year || "N/A"}</Badge>
          <Badge variant="secondary">Funding: {formatFundingRatio(chunk.funding_ratio)}</Badge>
          <Badge variant="secondary">Quality: {formatQuality(chunk.quality)}</Badge>
        </div>
      </CardContent>
    </Card>
  );
}
