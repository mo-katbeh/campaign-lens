import { Bot, FileSearch } from "lucide-react";
import { useState } from "react";

import { ChunkCard } from "@/components/dashboard/ChunkCard";
import { EmptyState } from "@/components/dashboard/EmptyState";
import { LoadingState } from "@/components/dashboard/LoadingState";
import { StatusAlert } from "@/components/dashboard/StatusAlert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { getAnswer } from "@/lib/api";
import { DEFAULT_TOP_K } from "@/lib/constants";
import type { AnswerResult, ApiFilters } from "@/lib/types";

interface AnswerTabProps {
  filters?: ApiFilters;
}

export function AnswerTab({ filters }: AnswerTabProps) {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AnswerResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    if (!question.trim()) {
      setError("Enter a question so the system can retrieve evidence and generate a grounded answer.");
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const response = await getAnswer({
        question: question.trim(),
        filters,
        top_k: DEFAULT_TOP_K,
      });
      setResult(response);
    } catch (requestError) {
      setResult(null);
      setError(requestError instanceof Error ? requestError.message : "Answer generation failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card className="border-border/80 bg-card/80 shadow-none">
        <CardHeader className="p-5">
          <CardTitle>Grounded Question Answering</CardTitle>
          <CardDescription>The answer is always paired with the retrieved evidence used to ground it.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 p-5 pt-0">
          <Textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about funding, campaign patterns, beneficiary groups, or gaps in support"
          />
          <div className="flex flex-wrap gap-3">
            <Button onClick={handleSubmit} disabled={isLoading}>
              <Bot className="h-4 w-4" />
              Get Answer
            </Button>
            <Badge variant="secondary">Endpoint: /answer</Badge>
            <Badge variant="outline">Grounded in retrieved chunks</Badge>
          </div>
        </CardContent>
      </Card>

      {error ? <StatusAlert message={error} /> : null}

      {isLoading ? <LoadingState title="Retrieving evidence and generating answer..." /> : null}

      {!isLoading && result ? (
        <section className="space-y-6">
          <Card className="border-primary/20 bg-primary/5 shadow-none">
            <CardHeader className="p-5">
              <div className="flex flex-wrap items-center gap-3">
                <Badge>Grounded answer</Badge>
                <Badge variant="secondary">Sources: {result.retrieved_chunks.length}</Badge>
              </div>
              <CardTitle className="text-lg">Answer</CardTitle>
              <CardDescription>This response is generated from the retrieved campaign data below.</CardDescription>
            </CardHeader>
            <CardContent className="p-5 pt-0">
              <p className="whitespace-pre-line text-sm leading-7 text-slate-100">{result.answer}</p>
            </CardContent>
          </Card>

          <section className="space-y-4">
            <div className="flex items-center gap-2">
              <FileSearch className="h-4 w-4 text-primary" />
              <h3 className="text-lg font-semibold">Sources</h3>
            </div>
            {result.retrieved_chunks.length > 0 ? (
              <div className="space-y-4">
                {result.retrieved_chunks.map((chunk) => (
                  <ChunkCard key={chunk.chunk_id} chunk={chunk} />
                ))}
              </div>
            ) : (
              <EmptyState
                title="No source chunks returned"
                description="The backend did not return supporting chunks for this answer."
              />
            )}
          </section>
        </section>
      ) : null}

      {!isLoading && !result && !error ? (
        <EmptyState
          title="Ask a grounded question"
          description="Answers will appear with a dedicated evidence section so you can inspect the supporting campaign chunks."
        />
      ) : null}
    </div>
  );
}
