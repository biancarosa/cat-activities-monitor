'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { 
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Copy, Check, FileText } from 'lucide-react';

interface JsonViewerModalProps {
  data: unknown;
  title?: string;
  triggerText?: string;
  className?: string;
}

export default function JsonViewerModal({ 
  data, 
  title = "JSON Data", 
  triggerText = "View JSON",
  className = "" 
}: JsonViewerModalProps) {
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
      <Dialog>
        <DialogTrigger asChild>
          <Button variant="ghost" size="sm" className="text-xs text-muted-foreground">
            <FileText className="h-3 w-3 mr-1" />
            {triggerText}
          </Button>
        </DialogTrigger>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <DialogTitle>{title}</DialogTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                title="Copy JSON"
              >
                {copied ? (
                  <>
                    <Check className="h-4 w-4 mr-1 text-green-600" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4 mr-1" />
                    Copy
                  </>
                )}
              </Button>
            </div>
          </DialogHeader>
          
          <div className="bg-muted/50 rounded-md p-4 border overflow-auto max-h-[60vh]">
            <pre className="text-sm font-mono whitespace-pre-wrap break-words">
              <code>{jsonString}</code>
            </pre>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
} 