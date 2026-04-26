import { SearchX } from "lucide-react";

interface EmptyStateProps {
  title: string;
  description: string;
}

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="flex min-h-[220px] flex-col items-center justify-center rounded-lg border border-dashed border-border bg-secondary/20 px-6 py-12 text-center">
      <SearchX className="mb-4 h-8 w-8 text-muted-foreground" />
      <h3 className="text-base font-semibold text-foreground">{title}</h3>
      <p className="mt-2 max-w-md text-sm text-muted-foreground">{description}</p>
    </div>
  );
}
