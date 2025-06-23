'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { BarChart3 } from 'lucide-react';
import ImageGallery from '@/components/ImageGallery';

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto p-8">
        {/* Quick Dashboard Access */}
        <div className="mb-6 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-white">Recent Detections</h1>
          <Button asChild className="flex items-center gap-2">
            <Link href="/dashboard">
              <BarChart3 className="h-4 w-4" />
              View Dashboard
            </Link>
          </Button>
        </div>
        
        <ImageGallery 
          className="w-full" 
        />
      </div>
    </div>
  );
}
