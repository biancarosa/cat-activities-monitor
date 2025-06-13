'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronRight, Copy, Check } from 'lucide-react';

interface JsonViewerProps {
  data: unknown;
  title?: string;
  className?: string;
}

export default function JsonViewer({ data, title = "JSON Data", className = "" }: JsonViewerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const jsonString = JSON.stringify(data, null, 2);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy JSON:', err);
    }
  };

  return (
    <div className={className}>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <div className="flex items-center justify-between">
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="p-0 h-auto font-normal">
              {isOpen ? (
                <ChevronDown className="h-3 w-3 mr-1" />
              ) : (
                <ChevronRight className="h-3 w-3 mr-1" />
              )}
              <span className="text-xs text-muted-foreground">{title}</span>
            </Button>
          </CollapsibleTrigger>
          
          {isOpen && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={handleCopy}
              title="Copy JSON"
            >
              {copied ? (
                <Check className="h-3 w-3 text-green-600" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </Button>
          )}
        </div>
        
        <CollapsibleContent className="mt-2">
          <div className="bg-muted/50 rounded-md p-3 border">
            <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap break-words max-h-96 overflow-y-auto">
              <code>{jsonString}</code>
            </pre>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
} 