'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  dashboardApi, 
  DashboardOverview, 
  CatActivitySummary, 
  LocationActivitySummary,
  ActivityTimeline 
} from '@/lib/api';
import { getCatColor } from '@/lib/colors';
import { Clock, MapPin, Activity, Users, TrendingUp, RefreshCw } from 'lucide-react';

export default function DashboardPage() {
  const [timeRange, setTimeRange] = useState<number>(24);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  
  // Dashboard data state
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [catsData, setCatsData] = useState<CatActivitySummary | null>(null);
  const [locationsData, setLocationsData] = useState<LocationActivitySummary | null>(null);
  const [timelineData, setTimelineData] = useState<ActivityTimeline | null>(null);

  const loadDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Load all dashboard data in parallel
      const [overviewData, catsResult, locationsResult, timelineResult] = await Promise.all([
        dashboardApi.getOverview(timeRange),
        dashboardApi.getCats(timeRange),
        dashboardApi.getLocations(timeRange),
        dashboardApi.getTimeline(timeRange, 'hour')
      ]);

      setOverview(overviewData);
      setCatsData(catsResult);
      setLocationsData(locationsResult);
      setTimelineData(timelineResult);
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const formatTimeAgo = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else {
      const diffDays = Math.floor(diffHours / 24);
      return `${diffDays}d ago`;
    }
  };

  const formatHour = (hour: number) => {
    if (hour === 0) return '12 AM';
    if (hour < 12) return `${hour} AM`;
    if (hour === 12) return '12 PM';
    return `${hour - 12} PM`;
  };

  // Activity color mapping
  const getActivityColor = (activity: string) => {
    const activityColors: Record<string, string> = {
      'sleeping': '#6366f1', // Indigo
      'eating': '#f59e0b',    // Amber
      'playing': '#10b981',   // Emerald
      'grooming': '#8b5cf6',  // Violet
      'sitting': '#06b6d4',   // Cyan
      'alert': '#ef4444',     // Red
      'walking': '#84cc16',   // Lime
      'drinking': '#3b82f6',  // Blue
      'perching': '#f97316',  // Orange
      'exploring': '#ec4899', // Pink
      'unknown': '#6b7280'    // Gray
    };
    return activityColors[activity] || '#6b7280';
  };

  // Get all unique cats from timeline data
  const getUniqueCats = (timelineData: ActivityTimeline) => {
    const cats = new Set<string>();
    timelineData.timeline.forEach(bucket => {
      Object.keys(bucket.named_cats).forEach(cat => cats.add(cat));
    });
    return Array.from(cats).sort();
  };

  // Get all unique activities from timeline data
  const getUniqueActivities = (timelineData: ActivityTimeline) => {
    const activities = new Set<string>();
    timelineData.timeline.forEach(bucket => {
      Object.keys(bucket.activities).forEach(activity => activities.add(activity));
    });
    return Array.from(activities).sort();
  };

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center py-12">
          <p className="text-red-500 mb-4">Error loading dashboard: {error}</p>
          <Button onClick={loadDashboardData}>Try Again</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white">Cat Activity Dashboard</h1>
          <p className="text-gray-400">
            Monitor your cats&apos; activities and locations in real-time
          </p>
          <p className="text-sm text-gray-500">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Select value={timeRange.toString()} onValueChange={(value) => setTimeRange(parseInt(value))}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1">Last 1 hour</SelectItem>
              <SelectItem value="6">Last 6 hours</SelectItem>
              <SelectItem value="24">Last 24 hours</SelectItem>
              <SelectItem value="72">Last 3 days</SelectItem>
              <SelectItem value="168">Last week</SelectItem>
            </SelectContent>
          </Select>
          
          <Button 
            onClick={loadDashboardData} 
            disabled={loading}
            size="sm"
            variant="outline"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </div>

      {loading && !overview ? (
        <div className="text-center py-12">
          <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-gray-400">Loading dashboard data...</p>
        </div>
      ) : (
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="cats">Cats</TabsTrigger>
            <TabsTrigger value="locations">Locations</TabsTrigger>
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            {overview && (
              <>
                {/* Overview Stats */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Named Cats</CardTitle>
                      <Users className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{overview.total_named_cats}</div>
                      <p className="text-xs text-muted-foreground">
                        {overview.named_cats_seen_recently} active recently
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Detections</CardTitle>
                      <Activity className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{overview.total_detections}</div>
                      <p className="text-xs text-muted-foreground">
                        {overview.total_cats_detected} cats detected
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Avg Cats/Detection</CardTitle>
                      <TrendingUp className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{overview.summary.avg_cats_per_detection}</div>
                      <p className="text-xs text-muted-foreground">
                        Detection efficiency
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Most Active Location</CardTitle>
                      <MapPin className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{overview.summary.most_active_location || 'N/A'}</div>
                      <p className="text-xs text-muted-foreground">
                        Primary activity zone
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Top Locations */}
                <Card>
                  <CardHeader>
                    <CardTitle>Top Active Locations</CardTitle>
                    <CardDescription>Locations with the most cat activity</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {overview.top_locations.map((location, index) => (
                        <div key={location.location} className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="text-lg font-bold text-gray-400">#{index + 1}</div>
                            <div>
                              <div className="font-medium">{location.location}</div>
                              <div className="text-sm text-gray-400">
                                {location.activity_count} cat sightings
                              </div>
                            </div>
                          </div>
                          <Badge variant="secondary">
                            {location.activity_count}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Recent Activities */}
                <Card>
                  <CardHeader>
                    <CardTitle>Recent Activities</CardTitle>
                    <CardDescription>Latest cat activities detected</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {overview.recent_activities.map((activity, index) => (
                        <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50">
                          <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2">
                              <Clock className="h-4 w-4 text-gray-400" />
                              <span className="text-sm text-gray-400">
                                {formatTimeAgo(activity.timestamp)}
                              </span>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            {activity.cat_name && (
                              <Badge 
                                style={{ backgroundColor: getCatColor(activity.cat_name) }}
                                className="text-white"
                              >
                                {activity.cat_name}
                              </Badge>
                            )}
                            <Badge variant="outline">{activity.location}</Badge>
                            {activity.activity && (
                              <Badge variant="secondary">{activity.activity}</Badge>
                            )}
                            <span className="text-sm text-gray-400">
                              {Math.round(activity.confidence * 100)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>

          <TabsContent value="cats" className="space-y-6">
            {catsData && (
              <>
                <div className="flex justify-between items-center">
                  <div>
                    <h2 className="text-2xl font-bold text-white">Cat Activity Summary</h2>
                    <p className="text-gray-400">
                      {catsData.active_cats} of {catsData.total_cats} cats active in the last {timeRange} hours
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {catsData.cats.map((cat) => (
                    <Card key={cat.cat_uuid} className={cat.is_active ? '' : 'opacity-60'}>
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="flex items-center gap-2">
                            <div 
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: getCatColor(cat.cat_name) }}
                            />
                            {cat.cat_name}
                          </CardTitle>
                          {cat.is_active ? (
                            <Badge className="bg-green-600">Active</Badge>
                          ) : (
                            <Badge variant="secondary">Inactive</Badge>
                          )}
                        </div>
                        {cat.description && (
                          <CardDescription>{cat.description}</CardDescription>
                        )}
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-400">Detections:</span>
                            <div className="font-medium">{cat.total_detections}</div>
                          </div>
                          <div>
                            <span className="text-gray-400">Avg Confidence:</span>
                            <div className="font-medium">{Math.round(cat.avg_confidence * 100)}%</div>
                          </div>
                          <div>
                            <span className="text-gray-400">Last Seen:</span>
                            <div className="font-medium">
                              {cat.last_seen ? formatTimeAgo(cat.last_seen) : 'Never'}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-400">Last Location:</span>
                            <div className="font-medium">{cat.last_location || 'Unknown'}</div>
                          </div>
                        </div>

                        {cat.favorite_locations.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium text-gray-400 mb-2">Favorite Locations</h4>
                            <div className="flex flex-wrap gap-2">
                              {cat.favorite_locations.slice(0, 3).map((location) => (
                                <Badge key={location.location} variant="outline" className="text-xs">
                                  {location.location} ({location.count})
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {cat.common_activities.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium text-gray-400 mb-2">Common Activities</h4>
                            <div className="flex flex-wrap gap-2">
                              {cat.common_activities.slice(0, 3).map((activity) => (
                                <Badge key={activity.activity} variant="secondary" className="text-xs">
                                  {activity.activity} ({activity.count})
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="locations" className="space-y-6">
            {locationsData && (
              <>
                <div className="flex justify-between items-center">
                  <div>
                    <h2 className="text-2xl font-bold text-white">Location Activity Summary</h2>
                    <p className="text-gray-400">
                      Activity across {locationsData.total_locations} monitored locations
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {locationsData.locations.map((location) => (
                    <Card key={location.location}>
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="flex items-center gap-2">
                            <MapPin className="h-5 w-5" />
                            {location.location}
                          </CardTitle>
                          <Badge variant="outline">
                            {location.total_cats_detected} cats
                          </Badge>
                        </div>
                        <CardDescription>
                          {location.total_detections} detections • {location.unique_cats_count} unique cats
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-400">Peak Hour:</span>
                            <div className="font-medium">{formatHour(location.peak_hour)}</div>
                          </div>
                          <div>
                            <span className="text-gray-400">Avg Confidence:</span>
                            <div className="font-medium">{Math.round(location.avg_confidence * 100)}%</div>
                          </div>
                          <div>
                            <span className="text-gray-400">Cats/Detection:</span>
                            <div className="font-medium">{location.avg_cats_per_detection}</div>
                          </div>
                          <div>
                            <span className="text-gray-400">Camera Status:</span>
                            <div className="font-medium">
                              {location.camera_config?.enabled ? 'Active' : 'Inactive'}
                            </div>
                          </div>
                        </div>

                        {location.unique_cats.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium text-gray-400 mb-2">Cats Detected</h4>
                            <div className="flex flex-wrap gap-2">
                              {location.unique_cats.map((catName) => (
                                <Badge 
                                  key={catName}
                                  style={{ backgroundColor: getCatColor(catName) }}
                                  className="text-white text-xs"
                                >
                                  {catName}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {location.common_activities.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium text-gray-400 mb-2">Common Activities</h4>
                            <div className="flex flex-wrap gap-2">
                              {location.common_activities.slice(0, 3).map((activity) => (
                                <Badge key={activity.activity} variant="secondary" className="text-xs">
                                  {activity.activity} ({activity.count})
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Simple hourly activity visualization */}
                        <div>
                          <h4 className="text-sm font-medium text-gray-400 mb-2">24-Hour Activity Pattern</h4>
                          <div className="flex items-end space-x-1 h-16">
                            {location.hourly_activity_pattern.map((count, hour) => (
                              <div
                                key={hour}
                                className="bg-blue-500 flex-1 min-w-[2px] rounded-t"
                                style={{
                                  height: `${Math.max(2, (count / Math.max(...location.hourly_activity_pattern)) * 100)}%`
                                }}
                                title={`${formatHour(hour)}: ${count} detections`}
                              />
                            ))}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="timeline" className="space-y-6">
            {timelineData && (
              <>
                <div className="flex justify-between items-center">
                  <div>
                    <h2 className="text-2xl font-bold text-white">Activity Timeline</h2>
                    <p className="text-gray-400">
                      {timelineData.summary.total_detections} detections over {timelineData.total_buckets} time periods
                    </p>
                  </div>
                </div>

                {/* Activity Legend */}
                <Card>
                  <CardHeader>
                    <CardTitle>Activity Color Legend</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {getUniqueActivities(timelineData).map((activity) => (
                        <Badge 
                          key={activity}
                          variant="outline" 
                          className="text-xs"
                          style={{ 
                            borderColor: getActivityColor(activity),
                            color: getActivityColor(activity)
                          }}
                        >
                          <div 
                            className="w-3 h-3 rounded-full mr-2"
                            style={{ backgroundColor: getActivityColor(activity) }}
                          />
                          {activity}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Per-Cat Activity Charts */}
                {getUniqueCats(timelineData).map((catName) => (
                  <Card key={catName}>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <div 
                          className="w-4 h-4 rounded-full"
                          style={{ backgroundColor: getCatColor(catName) }}
                        />
                        {catName} Activity Timeline
                      </CardTitle>
                      <CardDescription>Activity patterns over time for {catName}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {/* Cat activity chart */}
                        <div className="flex items-end space-x-1 h-32">
                          {timelineData.timeline.map((bucket, index) => {
                            const catActivities = bucket.cat_activities[catName] || {};
                            const totalCatActivity = Object.values(catActivities).reduce((sum, count) => sum + count, 0);
                            const maxActivity = Math.max(...timelineData.timeline.map(b => {
                              const activities = b.cat_activities[catName] || {};
                              return Object.values(activities).reduce((sum, count) => sum + count, 0);
                            }));
                            
                            return (
                              <div
                                key={index}
                                className="flex-1 min-w-[8px] flex flex-col justify-end"
                                title={`${new Date(bucket.timestamp).toLocaleString()}: ${totalCatActivity} activities`}
                              >
                                {/* Stacked activity bars */}
                                {Object.entries(catActivities).map(([activity, count], actIndex) => {
                                  const height = maxActivity > 0 ? (count / maxActivity) * 100 : 0;
                                  return (
                                    <div
                                      key={actIndex}
                                      className="w-full rounded-t"
                                      style={{
                                        backgroundColor: getActivityColor(activity),
                                        height: `${Math.max(2, height)}%`,
                                        marginTop: actIndex > 0 ? '1px' : '0'
                                      }}
                                      title={`${activity}: ${count}`}
                                    />
                                  );
                                })}
                              </div>
                            );
                          })}
                        </div>
                        
                        {/* Time labels */}
                        <div className="flex justify-between text-xs text-gray-400">
                          <span>{new Date(timelineData.timeline[0]?.timestamp).toLocaleString()}</span>
                          <span>{new Date(timelineData.timeline[timelineData.timeline.length - 1]?.timestamp).toLocaleString()}</span>
                        </div>
                        
                        {/* Cat's top activities */}
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(
                            timelineData.timeline.reduce((acc, bucket) => {
                              const catActivities = bucket.cat_activities[catName] || {};
                              Object.entries(catActivities).forEach(([activity, count]) => {
                                acc[activity] = (acc[activity] || 0) + count;
                              });
                              return acc;
                            }, {} as Record<string, number>)
                          )
                          .sort(([,a], [,b]) => b - a)
                          .slice(0, 5)
                          .map(([activity, count]) => (
                            <Badge 
                              key={activity}
                              variant="secondary" 
                              className="text-xs"
                              style={{ backgroundColor: getActivityColor(activity) + '20' }}
                            >
                              {activity}: {count}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}

                {/* Recent timeline entries */}
                <Card>
                  <CardHeader>
                    <CardTitle>Timeline Details</CardTitle>
                    <CardDescription>Detailed activity breakdown</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {timelineData.timeline.slice(-10).reverse().map((bucket, index) => (
                        <div key={index} className="p-3 rounded-lg bg-gray-800/50">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium">
                              {new Date(bucket.timestamp).toLocaleString()}
                            </span>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">{bucket.total_cats} cats</Badge>
                              <Badge variant="secondary">{bucket.total_detections} detections</Badge>
                            </div>
                          </div>
                          
                          {bucket.most_active_location && (
                            <div className="text-sm text-gray-400">
                              Most active: {bucket.most_active_location}
                              {bucket.primary_activity && ` • ${bucket.primary_activity}`}
                            </div>
                          )}
                          
                          {Object.keys(bucket.named_cats).length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {Object.entries(bucket.named_cats).map(([catName, count]) => (
                                <Badge 
                                  key={catName}
                                  style={{ backgroundColor: getCatColor(catName) }}
                                  className="text-white text-xs"
                                >
                                  {catName} ({count})
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}