'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Brain, 
  Database, 
  Target, 
  Users, 
  Zap,
  RefreshCw
} from 'lucide-react';
import { feedbackApi } from '@/lib/api';
import type { FeedbackListResponse } from '@/lib/api';

interface FeedbackProgressProps {
  totalImages: number;
  labeledImages: number;
  onRefresh?: () => void;
}

export default function FeedbackProgress({ 
  totalImages, 
  labeledImages, 
  onRefresh 
}: FeedbackProgressProps) {
  const [feedbackData, setFeedbackData] = useState<FeedbackListResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchFeedbackStats = async () => {
    try {
      setLoading(true);
      const response = await feedbackApi.list();
      setFeedbackData(response);
    } catch (error) {
      console.error('Failed to fetch feedback stats:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (totalImages > 0) {
      fetchFeedbackStats();
    }
  }, [totalImages, labeledImages]);

  const progressPercentage = totalImages > 0 ? (labeledImages / totalImages) * 100 : 0;
  
  const getProgressColor = (percentage: number) => {
    if (percentage >= 80) return 'bg-green-500';
    if (percentage >= 50) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getProgressMessage = (percentage: number) => {
    if (percentage >= 80) return 'Excellent! Great training data coverage';
    if (percentage >= 50) return 'Good progress! Keep labeling for better AI';
    if (percentage >= 20) return 'Getting started! More labels needed';
    return 'Just started - help train the AI by labeling images';
  };

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Brain className="h-5 w-5" />
          <span>AI Training Progress</span>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => {
              fetchFeedbackStats();
              onRefresh?.();
            }}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Images Labeled</span>
              <span className="font-medium">{labeledImages} / {totalImages}</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-500 ${getProgressColor(progressPercentage)}`}
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {getProgressMessage(progressPercentage)}
            </p>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center justify-center space-x-1 text-lg font-bold text-primary">
                <Database className="h-4 w-4" />
                <span>{Math.round(progressPercentage)}%</span>
              </div>
              <p className="text-xs text-muted-foreground">Completion</p>
            </div>
            
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center justify-center space-x-1 text-lg font-bold text-green-600">
                <Target className="h-4 w-4" />
                <span>{labeledImages}</span>
              </div>
              <p className="text-xs text-muted-foreground">Labeled</p>
            </div>
            
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center justify-center space-x-1 text-lg font-bold text-orange-600">
                <Zap className="h-4 w-4" />
                <span>{totalImages - labeledImages}</span>
              </div>
              <p className="text-xs text-muted-foreground">Remaining</p>
            </div>
            
            <div className="text-center p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center justify-center space-x-1 text-lg font-bold text-blue-600">
                <Users className="h-4 w-4" />
                <span>{feedbackData?.total_feedback || 0}</span>
              </div>
              <p className="text-xs text-muted-foreground">Total Feedback</p>
            </div>
          </div>

          {/* Motivation Messages */}
          <div className="text-center p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 rounded-lg border">
            {progressPercentage < 10 ? (
              <div className="space-y-2">
                <p className="text-sm font-medium text-blue-700 dark:text-blue-300">
                  🚀 Start Training the AI!
                </p>
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  Click the target icon (🎯) on any image to start labeling cats and their activities.
                </p>
              </div>
            ) : progressPercentage < 50 ? (
              <div className="space-y-2">
                <p className="text-sm font-medium text-yellow-700 dark:text-yellow-300">
                  📈 Great Start! Keep Going!
                </p>
                <p className="text-xs text-yellow-600 dark:text-yellow-400">
                  You&apos;re making progress! More labeled images = smarter AI detection.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-sm font-medium text-green-700 dark:text-green-300">
                  🎉 Amazing Work!
                </p>
                <p className="text-xs text-green-600 dark:text-green-400">
                  You&apos;ve created excellent training data. The AI is getting smarter with every label!
                </p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}