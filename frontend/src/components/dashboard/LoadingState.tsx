import { Loader2 } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";

interface LoadingStateProps {
  title: string;
  count?: number;
}

export function LoadingState({ title, count = 3 }: LoadingStateProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>{title}</span>
      </div>
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="rounded-lg border border-border bg-card/60 p-5">
          <Skeleton className="h-5 w-48" />
          <Skeleton className="mt-4 h-4 w-full" />
          <Skeleton className="mt-2 h-4 w-[90%]" />
          <div className="mt-4 flex gap-2">
            <Skeleton className="h-5 w-20" />
            <Skeleton className="h-5 w-24" />
            <Skeleton className="h-5 w-16" />
          </div>
        </div>
      ))}
    </div>
  );
}
