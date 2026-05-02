import { Menu, Radar, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";

import { AnswerTab } from "@/components/dashboard/AnswerTab";
import { RecommendationsTab } from "@/components/dashboard/RecommendationsTab";
import { SearchTab } from "@/components/dashboard/SearchTab";
import { SidebarFilters } from "@/components/dashboard/SidebarFilters";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DEFAULT_FILTER_DRAFT } from "@/lib/constants";
import { buildApiFilters, countActiveFilters } from "@/lib/filters";
import type { FilterDraft } from "@/lib/types";

type ActiveTab = "search" | "answer" | "recommendations";

export default function App() {
  const [draftFilters, setDraftFilters] = useState<FilterDraft>(DEFAULT_FILTER_DRAFT);
  const [appliedFilters, setAppliedFilters] = useState(() => buildApiFilters(DEFAULT_FILTER_DRAFT));
  const [isMobileFiltersOpen, setIsMobileFiltersOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<ActiveTab>("search");

  const activeFilterCount = useMemo(() => countActiveFilters(appliedFilters), [appliedFilters]);

  function applyFilters() {
    setAppliedFilters(buildApiFilters(draftFilters));
  }

  function applyFiltersAndClose() {
    applyFilters();
    setIsMobileFiltersOpen(false);
  }

  return (
    <div className="min-h-screen">
      <div className="panel-grid">
        <div className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8">
          <header className="mb-6 rounded-lg border border-border/70 bg-card/70 px-5 py-5 shadow-none backdrop-blur">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="rounded-md border border-primary/20 bg-primary/10 p-2 text-primary">
                    <Radar className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-xs font-medium uppercase tracking-[0.22em] text-primary/80">CampaignLens</p>
                    <h1 className="text-2xl font-semibold tracking-tight text-foreground">Retrieval-Augmented Campaign Intelligence</h1>
                  </div>
                </div>
                <p className="max-w-3xl text-sm leading-6 text-muted-foreground">
                  Explore retrieval evidence, inspect grounded answers, and compare similar campaigns with filters that shape the full RAG workflow.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <Badge variant="secondary">RAG system</Badge>
                <Badge variant="secondary">Transparent sources</Badge>
                <Badge variant="outline">{activeFilterCount} active filters</Badge>
                <Sheet open={isMobileFiltersOpen} onOpenChange={setIsMobileFiltersOpen}>
                  <SheetTrigger asChild>
                    <Button variant="outline" size="sm" className="lg:hidden">
                      <Menu className="h-4 w-4" />
                      Filters
                    </Button>
                  </SheetTrigger>
                  <SheetContent>
                    <SheetHeader>
                      <SheetTitle>Filter campaigns</SheetTitle>
                      <SheetDescription>Apply filters once and reuse them across search, answers, and recommendations.</SheetDescription>
                    </SheetHeader>
                    <div className="mt-6 flex-1">
                      <SidebarFilters draftFilters={draftFilters} onDraftChange={setDraftFilters} onApply={applyFiltersAndClose} />
                    </div>
                  </SheetContent>
                </Sheet>
              </div>
            </div>
          </header>

          <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
            <aside className="hidden lg:block">
              <SidebarFilters draftFilters={draftFilters} onDraftChange={setDraftFilters} onApply={applyFilters} />
            </aside>

            <main className="min-w-0">
              <section className="rounded-lg border border-border/70 bg-card/60 p-5 backdrop-blur">
                <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <h2 className="text-lg font-semibold">RAG Workspace</h2>
                    <p className="text-sm text-muted-foreground">
                      Each view exposes retrieval evidence directly so users can trace outputs back to campaign data.
                    </p>
                  </div>
                  <div className="inline-flex items-center gap-2 rounded-md border border-primary/20 bg-primary/10 px-3 py-2 text-xs text-primary">
                    <Sparkles className="h-3.5 w-3.5" />
                    Current view: {activeTab === "search" ? "Search" : activeTab === "answer" ? "Ask Question" : "Recommendations"}
                  </div>
                </div>

                <Separator className="mb-5" />

                <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as ActiveTab)} className="w-full">
                  <TabsList className="grid h-auto w-full grid-cols-1 gap-1 bg-transparent p-0 sm:w-auto sm:grid-cols-3 sm:rounded-md sm:border sm:border-border sm:bg-secondary/70 sm:p-1">
                    <TabsTrigger value="search" className="justify-start sm:justify-center">
                      Search
                    </TabsTrigger>
                    <TabsTrigger value="answer" className="justify-start sm:justify-center">
                      Ask Question
                    </TabsTrigger>
                    {/* <TabsTrigger value="recommendations" className="justify-start sm:justify-center">
                      Recommendations
                    </TabsTrigger> */}
                  </TabsList>

                  <TabsContent value="search">
                    <SearchTab filters={appliedFilters} />
                  </TabsContent>
                  <TabsContent value="answer">
                    <AnswerTab filters={appliedFilters} />
                  </TabsContent>
                  <TabsContent value="recommendations">
                    <RecommendationsTab filters={appliedFilters} />
                  </TabsContent>
                </Tabs>
              </section>
            </main>
          </div>
        </div>
      </div>
    </div>
  );
}
