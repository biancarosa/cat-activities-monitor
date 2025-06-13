'use client';

import ImageGallery from '@/components/ImageGallery';

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto p-8">
        <ImageGallery 
          className="w-full" 
        />
      </div>
    </div>
  );
}
