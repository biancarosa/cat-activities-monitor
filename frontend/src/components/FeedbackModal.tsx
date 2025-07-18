'use client';

import { useState, useEffect } from 'react';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription,
  DialogFooter,
  DialogTrigger 
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { 
  Tag, 
  Save, 
  AlertCircle,
  CheckCircle,
  Cat,
  MessageSquare,
  Check,
  X,
  Edit3,
  SkipForward
} from 'lucide-react';
import { DetectionImage, FeedbackAnnotation, ImageFeedback, feedbackApi, catProfileApi, CatProfile } from '@/lib/api';
import { getCatColor, getCatColorLight } from '@/lib/colors';
import { configManager } from '@/lib/config';
import Image from 'next/image';

interface FeedbackModalProps {
  image: DetectionImage;
  onFeedbackSubmitted?: () => void;
  trigger?: React.ReactElement;
}

export default function FeedbackModal({ 
  image, 
  onFeedbackSubmitted,
  trigger 
}: FeedbackModalProps) {
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [catProfiles, setCatProfiles] = useState<CatProfile[]>([]);
  
  // Form state for each detected cat
  const [catAnnotations, setCatAnnotations] = useState<Record<number, {
    catName: string;
    activity: string;
    activityFeedback: string;
    confidence: number;
    feedbackType: 'confirm' | 'reject' | 'correct' | 'skip';
    catProfileUuid: string;
    correctedClassId?: number;
    correctedClassName?: string;
  }>>({});
  
  const [notes, setNotes] = useState('');

  const handleCatAnnotationChange = (detectionIndex: number, field: string, value: string | number | undefined) => {
    setCatAnnotations(prev => ({
      ...prev,
      [detectionIndex]: {
        ...prev[detectionIndex],
        [field]: value
      }
    }));
  };

  // Auto-determine overall feedback type based on individual cat feedback
  const getOverallFeedbackType = (): 'correction' | 'addition' | 'confirmation' | 'rejection' => {
    const catFeedbackTypes = Object.values(catAnnotations).map(annotation => annotation?.feedbackType).filter(Boolean);
    
    if (catFeedbackTypes.length === 0) return 'correction';
    
    const hasReject = catFeedbackTypes.includes('reject');
    const hasCorrect = catFeedbackTypes.includes('correct');
    const hasConfirm = catFeedbackTypes.includes('confirm');
    
    // If any cats are rejected, it's a correction (mixed feedback)
    if (hasReject && (hasCorrect || hasConfirm)) return 'correction';
    
    // If all cats are rejected, it's a rejection
    if (hasReject && !hasCorrect && !hasConfirm) return 'rejection';
    
    // If any cats need correction, it's a correction
    if (hasCorrect) return 'correction';
    
    // If all cats are confirmed, it's a confirmation
    if (hasConfirm && !hasCorrect && !hasReject) return 'confirmation';
    
    return 'correction';
  };

  const handleSubmit = async () => {
    try {
      setSubmitting(true);
      setError(null);

      // Create user annotations from the form data
      const userAnnotations: FeedbackAnnotation[] = [];
      
      image.detections.forEach((detection, index) => {
        const annotation = catAnnotations[index];
        
        // Only include cats that have been given explicit feedback (not 'skip')
        if (annotation && annotation.feedbackType && annotation.feedbackType !== 'skip') {
          // For rejected cats, we still include them but with minimal annotation
          // For confirmed/corrected cats, we include full annotations
          userAnnotations.push({
            class_id: detection.class_id,
            class_name: detection.class_name,
            bounding_box: detection.bounding_box,
            confidence: annotation.confidence || detection.confidence,
            cat_name: annotation.catName || undefined,
            activity_feedback: annotation.activityFeedback || undefined,
            correct_activity: annotation.activity !== 'unknown' ? annotation.activity : undefined,
            activity_confidence: annotation.confidence || 0.8,
            cat_profile_uuid: annotation.catProfileUuid || undefined,
            corrected_class_id: annotation.correctedClassId || undefined,
            corrected_class_name: annotation.correctedClassName || undefined
          });
        }
      });

      // Determine overall feedback type automatically
      const determinedFeedbackType = getOverallFeedbackType();

      // Validation logic
      const hasAnyFeedback = Object.values(catAnnotations).some(annotation => 
        annotation?.feedbackType && annotation.feedbackType !== 'skip'
      );
      
      if (!hasAnyFeedback) {
        setError('Please provide feedback for at least one cat (confirm, reject, or correct)');
        return;
      }
      
      if (determinedFeedbackType === 'rejection' && !notes.trim()) {
        setError('Please provide a note explaining why all detections are rejected');
        return;
      }

      const feedbackData: ImageFeedback = {
        image_filename: image.filename,
        image_path: `/detections/${image.filename}`,
        original_detections: image.detections.map(d => ({
          class_id: d.class_id,
          class_name: d.class_name,
          confidence: d.confidence,
          bounding_box: d.bounding_box
        })),
        user_annotations: userAnnotations,
        feedback_type: determinedFeedbackType,
        notes: notes || undefined,
        timestamp: new Date().toISOString(),
        user_id: 'web_user'
      };

      await feedbackApi.submit(feedbackData);
      
      setSuccess(true);
      setTimeout(() => {
        setOpen(false);
        setSuccess(false);
        setCatAnnotations({});
        setNotes('');
        onFeedbackSubmitted?.();
      }, 2000);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  const getImageUrl = (filename: string) => {
    return `${configManager.getApiUrl()}/static/${filename}`;
  };

  const defaultTrigger = (
    <Button variant="outline" size="sm">
      <MessageSquare className="h-4 w-4 mr-1" />
      Label Cats
    </Button>
  );

  useEffect(() => {
    if (open) {
      catProfileApi.list().then(res => setCatProfiles(res.cats)).catch(() => setCatProfiles([]));
    }
  }, [open]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger || defaultTrigger}
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Tag className="h-5 w-5" />
            <span>Feedback</span>
          </DialogTitle>
          <DialogDescription>
            🧠 <strong>Help train the AI!</strong> Give individual feedback for each cat detection - confirm correct ones, 
            reject false positives, or correct inaccuracies. Your feedback will improve future detection accuracy.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <div className="p-3 bg-destructive/10 border border-destructive/50 rounded-md">
            <div className="flex items-center space-x-2 text-destructive">
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{error}</span>
            </div>
          </div>
        )}

        {success && (
          <div className="p-3 bg-green-50 border border-green-200 rounded-md dark:bg-green-950/20 dark:border-green-800">
            <div className="flex items-center space-x-2 text-green-700 dark:text-green-400">
              <CheckCircle className="h-4 w-4" />
              <span className="text-sm">
                🎉 Feedback submitted!
              </span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Image Preview */}
          <div className="space-y-4">
            <div className="aspect-video relative bg-muted rounded-lg overflow-hidden">
              <Image
                src={getImageUrl(image.filename)}
                alt={`Detection from ${image.source}`}
                className="w-full h-full object-cover"
                width={640}
                height={360}
                unoptimized
              />
              <div className="absolute top-2 left-2 flex flex-col space-y-1">
                <Badge variant="secondary" className="text-xs">
                  {image.source}
                </Badge>
                <Badge variant="default" className="text-xs flex items-center space-x-1">
                  {image.activities_by_cat && Object.keys(image.activities_by_cat).length > 0 && (
                    <div className="flex space-x-1 mr-1">
                      {Object.keys(image.activities_by_cat).slice(0, 3).map((catIndex) => {
                        const catIndexNum = parseInt(catIndex);
                        const catColor = getCatColor(undefined, undefined, catIndexNum);
                        return (
                          <div
                            key={catIndex}
                            className="h-2 w-2 rounded-full border"
                            style={{ 
                              backgroundColor: catColor,
                              borderColor: 'white'
                            }}
                          />
                        );
                      })}
                      {Object.keys(image.activities_by_cat).length > 3 && (
                        <span className="text-xs">+{Object.keys(image.activities_by_cat).length - 3}</span>
                      )}
                    </div>
                  )}
                  <span>{image.cat_count} cat{image.cat_count > 1 ? 's' : ''} detected</span>
                </Badge>
              </div>
            </div>
            
            <div className="text-sm text-muted-foreground">
              <p><strong>Timestamp:</strong> {image.timestamp_display}</p>
              <p><strong>Confidence:</strong> {image.max_confidence ? `${(image.max_confidence * 100).toFixed(0)}%` : 'N/A'}</p>
            </div>
          </div>

          {/* Feedback Form */}
          <div className="space-y-6">
            {/* Overall Feedback Status */}
            <div className="space-y-2">
              <Label>Overall Feedback Status</Label>
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-sm text-muted-foreground">
                  🤖 <strong>Auto-determined:</strong> {getOverallFeedbackType().charAt(0).toUpperCase() + getOverallFeedbackType().slice(1)}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  This is automatically calculated based on your per-cat feedback below.
                </p>
              </div>
            </div>

            {/* Cat Annotations */}
            <div className="space-y-4">
              <Label className="flex items-center space-x-2">
                <Cat className="h-4 w-4" />
                <span>Cat Labels ({image.detections.length} detected)</span>
              </Label>
              
              {image.detections.map((detection, index) => {
                const currentFeedback = catAnnotations[index]?.feedbackType || 'skip';
                const catColor = getCatColor(undefined, undefined, index);
                const catColorLight = getCatColorLight(undefined, undefined, index);
                
                return (
                <div key={index} className={`p-4 border rounded-lg space-y-3 ${
                  currentFeedback === 'confirm' ? 'border-green-500 bg-green-50 dark:bg-green-950/20' :
                  currentFeedback === 'reject' ? 'border-red-500 bg-red-50 dark:bg-red-950/20' :
                  currentFeedback === 'correct' ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20' :
                  'border-gray-200'
                }`}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div 
                        className="h-4 w-4 rounded-full border-2 flex-shrink-0"
                        style={{ 
                          backgroundColor: catColorLight,
                          borderColor: catColor 
                        }}
                      />
                      <h4 className="font-medium" style={{ color: catColor }}>
                        Cat {index + 1}
                      </h4>
                      <span className="ml-2 text-xs text-muted-foreground">({detection.class_name})</span>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {(detection.confidence * 100).toFixed(0)}% confidence
                    </Badge>
                  </div>
                  
                  {/* Per-Cat Feedback Actions */}
                  <div className="space-y-2">
                    <Label className="text-xs">Your Feedback</Label>
                    <div className="flex gap-1">
                      <Button
                        size="sm"
                        variant={currentFeedback === 'confirm' ? 'default' : 'outline'}
                        onClick={() => handleCatAnnotationChange(index, 'feedbackType', 'confirm')}
                        className={`flex-1 ${currentFeedback === 'confirm' ? '' : 'border'}`}
                        style={currentFeedback !== 'confirm' ? { borderColor: catColor + '40' } : {}}
                      >
                        <Check className="h-3 w-3 mr-1" />
                        Confirm
                      </Button>
                      <Button
                        size="sm"
                        variant={currentFeedback === 'reject' ? 'destructive' : 'outline'}
                        onClick={() => handleCatAnnotationChange(index, 'feedbackType', 'reject')}
                        className={`flex-1 ${currentFeedback === 'reject' ? '' : 'border'}`}
                        style={currentFeedback !== 'reject' ? { borderColor: catColor + '40' } : {}}
                      >
                        <X className="h-3 w-3 mr-1" />
                        Reject
                      </Button>
                      <Button
                        size="sm"
                        variant={currentFeedback === 'correct' ? 'default' : 'outline'}
                        onClick={() => handleCatAnnotationChange(index, 'feedbackType', 'correct')}
                        className={`flex-1 ${currentFeedback === 'correct' ? '' : 'border'}`}
                        style={currentFeedback !== 'correct' ? { borderColor: catColor + '40' } : {}}
                      >
                        <Edit3 className="h-3 w-3 mr-1" />
                        Correct
                      </Button>
                      <Button
                        size="sm"
                        variant={currentFeedback === 'skip' ? 'secondary' : 'outline'}
                        onClick={() => handleCatAnnotationChange(index, 'feedbackType', 'skip')}
                        className={`flex-1 ${currentFeedback === 'skip' ? '' : 'border'}`}
                        style={currentFeedback !== 'skip' ? { borderColor: catColor + '40' } : {}}
                      >
                        <SkipForward className="h-3 w-3 mr-1" />
                        Skip
                      </Button>
                    </div>
                    
                    {/* Feedback Status */}
                    {currentFeedback !== 'skip' && (
                      <div className="text-xs text-center">
                        {currentFeedback === 'confirm' && '✅ Marking this cat detection as correct'}
                        {currentFeedback === 'reject' && '❌ Marking this cat detection as wrong (false positive)'}
                        {currentFeedback === 'correct' && '✏️ Will apply corrections to this cat detection'}
                      </div>
                    )}
                  </div>
                  
                  {/* Detailed Annotations - Only show for confirm/correct */}
                  {(currentFeedback === 'confirm' || currentFeedback === 'correct') && (
                    <div className="grid grid-cols-1 gap-3 border-t pt-3">
                      <Label className="text-xs font-medium flex items-center space-x-2">
                        <div 
                          className="h-2 w-2 rounded-full"
                          style={{ backgroundColor: catColor }}
                        />
                        <span>Cat Profile</span>
                      </Label>
                      <select
                        className="w-full px-3 py-2 text-sm border rounded-md bg-background"
                        value={catAnnotations[index]?.catProfileUuid || ''}
                        onChange={e => handleCatAnnotationChange(index, 'catProfileUuid', e.target.value)}
                      >
                        <option value="">Select a cat profile...</option>
                        {catProfiles.map(profile => (
                          <option key={profile.cat_uuid} value={profile.cat_uuid}>{profile.name}</option>
                        ))}
                      </select>
                    </div>
                  )}
                  
                  {/* Class Correction - Only show for correct */}
                  {currentFeedback === 'correct' && (
                    <div className="grid grid-cols-1 gap-3 border-t pt-3">
                      <Label className="text-xs font-medium flex items-center space-x-2">
                        <div 
                          className="h-2 w-2 rounded-full"
                          style={{ backgroundColor: catColor }}
                        />
                        <span>Correct Classification</span>
                      </Label>
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">
                          Currently detected as: <strong>{detection.class_name}</strong>
                        </p>
                        <select
                          className="w-full px-3 py-2 text-sm border rounded-md bg-background"
                          value={catAnnotations[index]?.correctedClassId || ''}
                          onChange={e => {
                            const value = e.target.value;
                            if (value === 'reject') {
                              handleCatAnnotationChange(index, 'correctedClassId', -1);
                              handleCatAnnotationChange(index, 'correctedClassName', 'reject');
                            } else if (value) {
                              handleCatAnnotationChange(index, 'correctedClassId', parseInt(value));
                              handleCatAnnotationChange(index, 'correctedClassName', value === '15' ? 'cat' : 'dog');
                            } else {
                              handleCatAnnotationChange(index, 'correctedClassId', undefined);
                              handleCatAnnotationChange(index, 'correctedClassName', undefined);
                            }
                          }}
                        >
                          <option value="">Keep original classification ({detection.class_name})</option>
                          <option value="15">🐱 Cat</option>
                          <option value="16">🐕 Dog</option>
                          <option value="reject">❌ Not an animal (reject)</option>
                        </select>
                        {catAnnotations[index]?.correctedClassId && catAnnotations[index]?.correctedClassId !== detection.class_id && (
                          <p className="text-xs text-blue-600 dark:text-blue-400">
                            ✏️ Will be corrected to: <strong>{catAnnotations[index]?.correctedClassName}</strong>
                          </p>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Rejection Reason - Only show for reject */}
                  {currentFeedback === 'reject' && (
                    <div className="border-t pt-3">
                      <Label className="text-xs text-red-600 dark:text-red-400">
                        ❌ This detection will be marked as incorrect (false positive)
                      </Label>
                    </div>
                  )}
                </div>
                );
              })}
            </div>

            {/* General Notes */}
            <div className="space-y-2">
              <Label htmlFor="notes">Additional Notes</Label>
              <textarea
                id="notes"
                className="w-full px-3 py-2 text-sm border rounded-md bg-background min-h-[80px]"
                placeholder="Any additional observations or feedback..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              />
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={submitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={submitting || success}>
            {submitting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Submitting...
              </>
            ) : success ? (
              <>
                <CheckCircle className="h-4 w-4 mr-2" />
                Submitted!
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-2" />
                Submit Feedback
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
} 