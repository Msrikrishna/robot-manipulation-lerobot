"use client";

import Image from "next/image";
import { Check, Bug } from "lucide-react";
import { cn } from "@/lib/utils";
import { STEPS } from "@/lib/wizard-types";
import { useWizard } from "./wizard-provider";
import { Button } from "@/components/ui/button";

export function WizardSidebar() {
  const { state, goToStep, dispatch } = useWizard();

  return (
    <aside className="fixed left-0 top-0 z-20 flex h-screen w-60 flex-col border-r bg-white">
      <div className="flex h-14 items-center border-b px-4">
        <Image
          src="/makermods-logo.png"
          alt="MakerMods"
          width={160}
          height={40}
          className="h-16 w-auto object-contain"
          priority
        />
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {STEPS.map((step, i) => {
          const isCurrent = i === state.currentStep;
          // A step only shows as complete if all its prerequisites are also complete
          // Steps 0-3 are sequential; steps 4-7 only require 0-3 (hardware setup)
          const prereqsMet =
            i <= 3
              ? state.completedSteps.slice(0, i).every(Boolean)
              : state.completedSteps.slice(0, 4).every(Boolean);
          const isComplete = state.completedSteps[i] && prereqsMet;

          return (
            <button
              key={i}
              onClick={() => {
                if (state.debugMode) dispatch({ type: "TOGGLE_DEBUG_MODE" });
                goToStep(i);
              }}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
                isCurrent && "bg-primary/5 font-medium text-foreground",
                !isCurrent &&
                  "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <StepIndicator
                step={i}
                isCurrent={isCurrent}
                isComplete={isComplete}
              />
              <span>{step.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="border-t px-4 py-3 flex items-center justify-between">
        <p className="text-xs text-muted-foreground">LeRobot Setup Wizard</p>
        <Button
          variant={state.debugMode ? "default" : "ghost"}
          size="icon"
          className="h-7 w-7"
          onClick={() => dispatch({ type: "TOGGLE_DEBUG_MODE" })}
          title="Hardware Diagnostics"
        >
          <Bug className="h-3.5 w-3.5" />
        </Button>
      </div>
    </aside>
  );
}

function StepIndicator({
  step,
  isCurrent,
  isComplete,
}: {
  step: number;
  isCurrent: boolean;
  isComplete: boolean;
}) {
  if (isComplete && !isCurrent) {
    return (
      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <Check className="h-3.5 w-3.5" />
      </div>
    );
  }

  if (isCurrent) {
    return (
      <div className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-primary">
        <div className="h-2 w-2 rounded-full bg-primary" />
      </div>
    );
  }

  return (
    <div className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-muted-foreground/30">
      <span className="text-xs text-muted-foreground">{step + 1}</span>
    </div>
  );
}
