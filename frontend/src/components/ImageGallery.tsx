'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Camera, 
  RefreshCw, 
  Eye, 
  Calendar, 
  AlertCircle,
  Image as ImageIcon,
  Download,
  RotateCcw,
  Loader2,
  CheckCircle,
  Star,
  Target
} from 'lucide-react';
import { detectionApi, cameraApi, DetectionImage } from '@/lib/api';
import { configManager } from '@/lib/config';
import JsonViewerModal from '@/components/JsonViewerModal';
import FeedbackModal from '@/components/FeedbackModal';
import { Pagination } from '@/components/ui/pagination';
import Image from 'next/image';

interface ImageGalleryProps {
  className?: string;
  onStatsUpdate?: (total: number, labeled: number) => void;
}

export default function ImageGallery({ className = '', onStatsUpdate }: ImageGalleryProps) {
  const [images, setImages] = useState<DetectionImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<string>('all');
  const [refreshing, setRefreshing] = useState(false);
  const [fetchingNew, setFetchingNew] = useState(false);
  const [reprocessingAll, setReprocessingAll] = useState(false);
  const [reprocessingImages, setReprocessingImages] = useState<Set<string>>(new Set());
  const [feedbackStats, setFeedbackStats] = useState<{total: number, annotatedImages: number} | null>(null);
  const [configLoaded, setConfigLoaded] = useState(() => configManager.isConfigLoaded());
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalImages, setTotalImages] = useState(0);
  const [pageSize] = useState(20);
  const currentPageRef = useRef(currentPage);
  const isMountedRef = useRef(true);

  // Update ref when currentPage changes
  useEffect(() => {
    currentPageRef.current = currentPage;
  }, [currentPage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const fetchImages = useCallback(async (page: number = currentPageRef.current) => {
    try {
      if (!isMountedRef.current) return;
      setLoading(true);
      setError(null);
      const response = await detectionApi.getImages(page, pageSize);
      
      if (!isMountedRef.current) return;
      setImages(response.images);
      setCurrentPage(response.page);
      setTotalPages(response.total_pages);
      setTotalImages(response.total);
      
      // Update feedback stats based on current page
      const annotatedCount = response.images.filter(img => img.has_feedback).length;
      setFeedbackStats({
        total: response.total,
        annotatedImages: annotatedCount
      });
      
      // Notify parent component of stats update (use total, not just current page)
      onStatsUpdate?.(response.total, annotatedCount);
    } catch (err) {
      if (!isMountedRef.current) return;
      setError(err instanceof Error ? err.message : 'Failed to fetch images');
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  }, [pageSize, onStatsUpdate]);

  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      await fetchImages(currentPageRef.current);
    } finally {
      setRefreshing(false);
    }
  };

  const handlePageChange = useCallback(async (page: number) => {
    setCurrentPage(page);
    await fetchImages(page);
  }, [fetchImages]);

  const handleFetchNew = async () => {
    try {
      setFetchingNew(true);
      await cameraApi.fetchAll();
      // Wait a moment for images to be processed, then refresh
      setTimeout(() => {
        fetchImages(currentPageRef.current);
      }, 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch new images');
    } finally {
      setFetchingNew(false);
    }
  };

  const handleReprocessImage = async (filename: string) => {
    try {
      setReprocessingImages(prev => new Set(prev).add(filename));
      setError(null);
      setSuccessMessage(null);
      
      const result = await detectionApi.reprocessImage(filename);
      
      // Show success message
      const detectionInfo = result.detection_result.detected 
        ? `${result.detection_result.count} cat${result.detection_result.count !== 1 ? 's' : ''} detected`
        : 'No cats detected';
      setSuccessMessage(`✅ Reprocessed ${filename}: ${detectionInfo}`);
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000);
      
      // Refresh the current page to show updated results
      await fetchImages(currentPageRef.current);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reprocess image');
    } finally {
      setReprocessingImages(prev => {
        const newSet = new Set(prev);
        newSet.delete(filename);
        return newSet;
      });
    }
  };

  const handleReprocessAll = async () => {
    try {
      setReprocessingAll(true);
      setError(null);
      setSuccessMessage(null);
      
      const result = await detectionApi.reprocessAllImages();
      
      // Show success message with detailed results
      setSuccessMessage(
        `✅ Bulk reprocessing completed: ${result.processed} images processed successfully` +
        (result.errors > 0 ? `, ${result.errors} errors` : '')
      );
      
      // Clear success message after 8 seconds for bulk operations
      setTimeout(() => setSuccessMessage(null), 8000);
      
      // Refresh the current page to show updated results
      await fetchImages(currentPageRef.current);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reprocess all images');
    } finally {
      setReprocessingAll(false);
    }
  };

  useEffect(() => {
    if (configLoaded && isMountedRef.current) {
      fetchImages(1);
      setCurrentPage(1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configLoaded, pageSize]);

  // Listen for config loaded event and API URL changes
  useEffect(() => {
    const handleConfigLoaded = () => {
      if (!isMountedRef.current) return;
      setConfigLoaded(true);
    };

    const handleApiUrlChanged = () => {
      // Force refresh images when API URL changes
      if (configLoaded && isMountedRef.current) {
        setCurrentPage(1);
        fetchImages(1);
      }
    };

    if (typeof window !== 'undefined') {
      window.addEventListener('configLoaded', handleConfigLoaded);
      window.addEventListener('apiUrlChanged', handleApiUrlChanged);
      return () => {
        window.removeEventListener('configLoaded', handleConfigLoaded);
        window.removeEventListener('apiUrlChanged', handleApiUrlChanged);
      };
    }
  }, [configLoaded, fetchImages]);

  // Filter images by source
  const filteredImages = selectedSource === 'all' 
    ? images 
    : images.filter(img => img.source === selectedSource);

  // Get unique sources for filter dropdown
  const sources = Array.from(new Set(images.map(img => img.source))).sort();

  const getImageUrl = (filename: string) => {
    // Images are served from /static static mount
    return `${configManager.getApiUrl()}/static/${filename}`;
  };

  const getConfidenceBadge = (confidence: number | null) => {
    if (!confidence) return null;
    
    if (confidence >= 0.8) {
      return <Badge variant="default" className="text-xs">High: {(confidence * 100).toFixed(0)}%</Badge>;
    } else if (confidence >= 0.5) {
      return <Badge variant="secondary" className="text-xs">Med: {(confidence * 100).toFixed(0)}%</Badge>;
    } else {
      return <Badge variant="outline" className="text-xs">Low: {(confidence * 100).toFixed(0)}%</Badge>;
    }
  };

  const getCatNames = (image: DetectionImage): string[] => {
    if (!image.detections) return [];
    
    return image.detections
      .map(detection => detection.cat_name || "unidentified")
      .filter((name, index, arr) => arr.indexOf(name) === index); // Remove duplicates
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="pt-6">
          <div className="flex items-center justify-center h-32">
            <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading images...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <ImageIcon className="h-5 w-5" />
            <div>
              <CardTitle className="flex items-center space-x-2">
                <span>Detection Images</span>
                {/* {feedbackStats && (
                  <Badge variant="secondary" className="text-xs">
                    <Database className="h-3 w-3 mr-1" />
                    {feedbackStats.annotatedImages}/{feedbackStats.total} labeled
                  </Badge>
                )} */}
              </CardTitle>
              <CardDescription>
                Recent images captured from cameras with AI detection results.
                {feedbackStats && feedbackStats.total > 0 && (
                  <span className="text-primary">
                    {' '}Help improve the AI by labeling more images.
                  </span>
                )}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <select
              value={selectedSource}
              onChange={(e) => setSelectedSource(e.target.value)}
              className="px-3 py-1 text-sm border rounded-md bg-background"
            >
              <option value="all">All Sources</option>
              {sources.map(source => (
                <option key={source} value={source}>{source}</option>
              ))}
            </select>
            <Button onClick={handleRefresh} variant="outline" size="sm" disabled={refreshing}>
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            </Button>
            <Button 
              onClick={handleReprocessAll} 
              variant="outline" 
              size="sm" 
              disabled={reprocessingAll || images.length === 0}
              title="Reprocess all images with current ML model"
            >
              {reprocessingAll ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4" />
              )}
              {reprocessingAll ? 'Reprocessing...' : 'Reprocess All'}
            </Button>
            <Button onClick={handleFetchNew} variant="default" size="sm" disabled={fetchingNew}>
              <Camera className="h-4 w-4 mr-1" />
              {fetchingNew ? 'Fetching...' : 'Fetch New'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="mb-4 p-3 bg-destructive/10 border border-destructive/50 rounded-md">
            <div className="flex items-center space-x-2 text-destructive">
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{error}</span>
            </div>
          </div>
        )}

        {successMessage && (
          <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-md dark:bg-green-950/20 dark:border-green-800">
            <div className="flex items-center space-x-2 text-green-700 dark:text-green-400">
              <CheckCircle className="h-4 w-4" />
              <span className="text-sm">{successMessage}</span>
            </div>
          </div>
        )}

        {filteredImages.length === 0 ? (
          <div className="text-center py-8">
            <ImageIcon className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium text-muted-foreground mb-2">No Images Found</h3>
            <p className="text-sm text-muted-foreground mb-4">
              {selectedSource === 'all' 
                ? 'No detection images available. Try fetching new images from cameras.'
                : `No images found for source "${selectedSource}".`
              }
            </p>
            <Button onClick={handleFetchNew} disabled={fetchingNew}>
              <Camera className="h-4 w-4 mr-2" />
              Fetch Images from Cameras
            </Button>
          </div>
        ) : (
          <>
            <div className="mb-4 flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Showing {filteredImages.length} of {totalImages} images
                {selectedSource !== 'all' && ` from ${selectedSource}`}
                {totalPages > 1 && ` (Page ${currentPage} of ${totalPages})`}
              </div>
              {totalPages > 1 && (
                <Pagination
                  currentPage={currentPage}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                  className="justify-end"
                />
              )}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredImages.map((image) => (
                <Card key={image.filename} className="overflow-hidden">
                  <div className="aspect-video relative bg-muted">
                    <Image
                      src={getImageUrl(image.filename)}
                      alt={`Detection from ${image.source}`}
                      className="w-full h-full object-cover"
                      width={640}
                      height={360}
                      unoptimized
                    />
                    <div className="hidden absolute inset-0 flex items-center justify-center bg-muted">
                      <div className="text-center">
                        <ImageIcon className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                        <p className="text-sm text-muted-foreground">Image not available</p>
                      </div>
                    </div>
                    
                    {/* Overlay badges */}
                    <div className="absolute top-2 left-2 flex flex-col space-y-1">
                      <Badge variant="secondary" className="text-xs">
                        {image.source}
                      </Badge>
                      {image.cat_count > 0 && (
                        <Badge variant="default" className="text-xs flex items-center space-x-1">
                          <span>{image.cat_count} cat{image.cat_count > 1 ? 's' : ''}</span>
                        </Badge>
                      )}
                      {getCatNames(image).map((catName, index) => (
                        <Badge 
                          key={index} 
                          variant="secondary" 
                          className={`text-xs ${
                            catName === "unidentified" 
                              ? "bg-gray-100 text-gray-700 border-gray-300" 
                              : "bg-green-100 text-green-800 border-green-300"
                          }`}
                        >
                          🐱 {catName}
                        </Badge>
                      ))}
                    </div>
                    
                    <div className="absolute top-2 right-2">
                      {getConfidenceBadge(image.max_confidence)}
                    </div>
                  </div>
                  
                  <CardContent className="p-3">
                    <div className="space-y-2">
                      <div className="flex items-center space-x-1 text-sm font-medium">
                        <Calendar className="h-3 w-3" />
                        <span>{image.timestamp_display}</span>
                      </div>
                      
                      <div className="space-y-1 text-xs text-muted-foreground">
                        <div className="flex items-center justify-between">
                          <span className="font-medium">Filename:</span>
                          <span className="truncate ml-2" title={image.filename}>
                            {image.filename}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="font-medium">Size:</span>
                          <span>{image.file_size_mb}MB</span>
                        </div>
                        {image.detections && image.detections.length > 0 && (
                          <div className="flex items-center justify-between">
                            <span className="font-medium">Cats:</span>
                            <span className="truncate ml-2">
                              {getCatNames(image).join(', ')}
                            </span>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex items-center justify-between pt-2">
                        <div className="flex items-center space-x-1">
                          {image.has_feedback ? (
                            <Badge variant="outline" className="text-xs border-green-500 text-green-700">
                              <Star className="h-3 w-3 mr-1" />
                              Labeled
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="text-xs border-orange-500 text-orange-700">
                              <Target className="h-3 w-3 mr-1" />
                              Needs Labeling
                            </Badge>
                          )}
                        </div>
                        
                        <div className="flex space-x-1">
                          <FeedbackModal 
                            image={image}
                            onFeedbackSubmitted={fetchImages}
                            trigger={
                              <Button 
                                size="sm" 
                                variant={!image.has_feedback ? "default" : "ghost"}
                                className={`h-6 w-6 p-0 ${!image.has_feedback ? 'bg-primary hover:bg-primary/90' : ''}`}
                                title={image.has_feedback ? "Update labeling" : "Label cats - Help improve AI!"}
                              >
                                {image.has_feedback ? (
                                  <CheckCircle className="h-3 w-3" />
                                ) : (
                                  <Target className="h-3 w-3" />
                                )}
                              </Button>
                            }
                          />
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            className="h-6 w-6 p-0"
                            onClick={() => handleReprocessImage(image.filename)}
                            disabled={reprocessingImages.has(image.filename)}
                            title="Reprocess this image with current ML model"
                          >
                            {reprocessingImages.has(image.filename) ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              <RotateCcw className="h-3 w-3" />
                            )}
                          </Button>
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            className="h-6 w-6 p-0"
                            onClick={() => window.open(getImageUrl(image.filename), '_blank')}
                          >
                            <Eye className="h-3 w-3" />
                          </Button>
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            className="h-6 w-6 p-0"
                            onClick={() => {
                              const link = document.createElement('a');
                              link.href = getImageUrl(image.filename);
                              link.download = image.filename;
                              link.click();
                            }}
                          >
                            <Download className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                      
                      {/* JSON Viewer Modal */}
                      <div className="pt-2 border-t mt-2">
                        <JsonViewerModal 
                          data={image} 
                          title="Image Data JSON"
                          triggerText="View Image Data JSON"
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            
            {totalPages > 1 && (
              <div className="mt-6 flex justify-center">
                <Pagination
                  currentPage={currentPage}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
} 