import { Filter, SlidersHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { YEARS, THEMES, QUALITY_MAX, QUALITY_MIN } from "@/lib/constants";
import type { FilterDraft } from "@/lib/types";

interface SidebarFiltersProps {
  draftFilters: FilterDraft;
  onDraftChange: (filters: FilterDraft) => void;
  onApply: () => void;
}

export function SidebarFilters({ draftFilters, onDraftChange, onApply }: SidebarFiltersProps) {
  return (
    <Card className="h-full border-border/80 bg-card/80 shadow-none lg:sticky lg:top-6">
      <CardHeader className="p-5">
        <div className="flex items-center gap-2">
          <div className="rounded-md border border-primary/20 bg-primary/10 p-2 text-primary">
            <Filter className="h-4 w-4" />
          </div>
          <div>
            <CardTitle>Filters</CardTitle>
            <CardDescription>Refine retrieval, grounded answers, and recommendations.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6 p-5 pt-0">
        <div className="space-y-2">
          <Label htmlFor="theme">Theme</Label>
          <Select value={draftFilters.theme} onValueChange={(value) => onDraftChange({ ...draftFilters, theme: value })}>
            <SelectTrigger id="theme">
              <SelectValue placeholder="All themes" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All themes</SelectItem>
              {THEMES.map((theme) => (
                <SelectItem key={theme} value={theme}>
                  {theme.replace(/_/g, " ")}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="year">Year</Label>
          <Select value={draftFilters.year} onValueChange={(value) => onDraftChange({ ...draftFilters, year: value })}>
            <SelectTrigger id="year">
              <SelectValue placeholder="All years" />
            </SelectTrigger>
            <SelectContent>
              {YEARS.map((year) => (
                <SelectItem key={year} value={year}>
                  {year === "all" ? "All years" : year}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label>Funding ratio</Label>
            <span className="text-xs text-muted-foreground">Min / Max</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Input
              inputMode="decimal"
              placeholder="0.0"
              value={draftFilters.fundingRatioMin}
              onChange={(event) => onDraftChange({ ...draftFilters, fundingRatioMin: event.target.value })}
            />
            <Input
              inputMode="decimal"
              placeholder="10.2"
              value={draftFilters.fundingRatioMax}
              onChange={(event) => onDraftChange({ ...draftFilters, fundingRatioMax: event.target.value })}
            />
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="quality">Quality score</Label>
            <span className="text-sm font-medium text-foreground">{draftFilters.qualityMin}</span>
          </div>
          <Slider
            id="quality"
            min={QUALITY_MIN}
            max={QUALITY_MAX}
            step={1}
            value={[draftFilters.qualityMin]}
            onValueChange={([value]) => onDraftChange({ ...draftFilters, qualityMin: value ?? QUALITY_MIN })}
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{QUALITY_MIN}</span>
            <span>{QUALITY_MAX}</span>
          </div>
        </div>

        <Button className="w-full" onClick={onApply}>
          <SlidersHorizontal className="h-4 w-4" />
          Apply Filters
        </Button>
      </CardContent>
    </Card>
  );
}
