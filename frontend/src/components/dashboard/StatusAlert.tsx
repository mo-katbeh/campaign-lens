import { AlertCircle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface StatusAlertProps {
  title?: string;
  message: string;
}

export function StatusAlert({ title = "Request failed", message }: StatusAlertProps) {
  return (
    <Alert>
      <div className="flex items-start gap-3">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
        <div>
          <AlertTitle>{title}</AlertTitle>
          <AlertDescription>{message}</AlertDescription>
        </div>
      </div>
    </Alert>
  );
}
