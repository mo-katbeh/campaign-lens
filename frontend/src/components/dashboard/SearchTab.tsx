import { Search } from "lucide-react";
import { useState } from "react";

import { ChunkCard } from "@/components/dashboard/ChunkCard";
import { EmptyState } from "@/components/dashboard/EmptyState";
import { LoadingState } from "@/components/dashboard/LoadingState";
import { StatusAlert } from "@/components/dashboard/StatusAlert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { retrieveChunks } from "@/lib/api";
import { DEFAULT_TOP_K } from "@/lib/constants";
import type { ApiFilters, RetrieveResponse } from "@/lib/types";

interface SearchTabProps {
  filters?: ApiFilters;
}

export function SearchTab({ filters }: SearchTabProps) {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RetrieveResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch() {
    if (!query.trim()) {
      setError("Enter a search query to retrieve campaign evidence.");
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const response = await retrieveChunks({
        query: query.trim(),
        filters,
        top_k: DEFAULT_TOP_K,
      });
      setResult(response);
    } catch (requestError) {
      setResult(null);
      setError(requestError instanceof Error ? requestError.message : "Search failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card className="border-border/80 bg-card/80 shadow-none">
        <CardHeader className="p-5">
          <CardTitle>Search Retrieval</CardTitle>
          <CardDescription>Inspect the exact chunks the retriever returns for your campaign query.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 p-5 pt-0">
          <div className="flex flex-col gap-3 md:flex-row">
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search campaign needs, themes, locations, or outcomes"
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  handleSearch();
                }
              }}
            />
            <Button className="md:w-36" onClick={handleSearch} disabled={isLoading}>
              <Search className="h-4 w-4" />
              Search
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">Endpoint: /retrieve</Badge>
            <Badge variant="secondary">Top K: {DEFAULT_TOP_K}</Badge>
            <Badge variant="outline">Transparent retrieval output</Badge>
          </div>
        </CardContent>
      </Card>

      {error ? <StatusAlert message={error} /> : null}

      {isLoading ? <LoadingState title="Retrieving campaign chunks..." /> : null}

      {!isLoading && result && result.chunks.length > 0 ? (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">Retrieved Chunks</h3>
              <p className="text-sm text-muted-foreground">{result.chunks.length} chunks matched the current query and filters.</p>
            </div>
          </div>
          <div className="space-y-4">
            {result.chunks.map((chunk) => (
              <ChunkCard key={chunk.chunk_id} chunk={chunk} />
            ))}
          </div>
        </section>
      ) : null}

      {!isLoading && result && result.chunks.length === 0 ? (
        <EmptyState
          title="No retrieval results"
          description="No chunks matched this search with the current filters. Try broadening the query or relaxing the filter set."
        />
      ) : null}

      {!isLoading && !result && !error ? (
        <EmptyState
          title="Run a retrieval search"
          description="Search results will appear here as chunk-level evidence with similarity scores and campaign metadata."
        />
      ) : null}
    </div>
  );
}
