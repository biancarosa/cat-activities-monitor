'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { 
  Settings, 
  Camera, 
  Server, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Monitor,
  Database,
  Cpu,
  Globe,
  Save,
  RotateCcw,
  Download,
  Brain,
  Play,
  Loader2
} from 'lucide-react';
import { systemApi, cameraApi, trainingApi, SystemConfig, SystemStatus, CamerasResponse, HealthStatus, TrainingStatus, apiClient } from '@/lib/api';
import { configManager } from '@/lib/config';

type BackendTrainingStatus = TrainingStatus & {
  current_model?: string;
  // Add any other backend fields you access
};

export default function SettingsPage() {
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [cameras, setCameras] = useState<CamerasResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [reloading, setReloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Training-specific state
  const [exportingData, setExportingData] = useState(false);
  const [retrainingModel, setRetrainingModel] = useState(false);
  const [exportSuccess, setExportSuccess] = useState<string | null>(null);
  const [retrainSuccess, setRetrainSuccess] = useState<string | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  
  // API URL configuration state
  const [apiUrl, setApiUrl] = useState(configManager.getApiUrl());
  const [tempApiUrl, setTempApiUrl] = useState(configManager.getApiUrl());
  const [savingApiUrl, setSavingApiUrl] = useState(false);
  const [apiUrlError, setApiUrlError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [configData, statusData, camerasData, healthData, trainingData] = await Promise.all([
        systemApi.getConfig().catch(() => null),
        systemApi.getStatus().catch(() => null),
        cameraApi.list().catch(() => null),
        systemApi.getHealth().catch(() => null),
        trainingApi.getStatus().catch(() => null),
      ]);

      setSystemConfig(configData);
      setSystemStatus(statusData);
      setCameras(camerasData);
      setHealthStatus(healthData);

      let mappedTrainingStatus = trainingData;
      if (trainingData && (!trainingData.stats || typeof trainingData.stats !== 'object')) {
        const backendData = trainingData as unknown as Record<string, unknown>;
        mappedTrainingStatus = {
          ...trainingData,
          stats: {
            total_feedback_entries: Number((backendData.feedback_data as Record<string, unknown>)?.total_feedback) || 0,
            total_annotations: Number((backendData.training_data as Record<string, unknown>)?.labels_count) || 0,
            named_cats: Number((backendData.feedback_data as Record<string, unknown>)?.cat_profiles) || 0,
          },
        };
      }
      setTrainingStatus(mappedTrainingStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  const handleReloadConfig = async () => {
    try {
      setReloading(true);
      await systemApi.reloadConfig();
      await fetchData(); // Refresh all data after reload
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reload configuration');
    } finally {
      setReloading(false);
    }
  };

  const handleSaveApiUrl = async () => {
    try {
      setSavingApiUrl(true);
      setApiUrlError(null);
      
      // Validate URL
      configManager.setApiUrl(tempApiUrl);
      
      // Update the API client
      apiClient.updateApiUrl(tempApiUrl);
      
      // Update local state
      setApiUrl(tempApiUrl);
      
      // Test the connection by fetching health status
      await systemApi.getHealth();
      
    } catch (err) {
      setApiUrlError(err instanceof Error ? err.message : 'Failed to save API URL');
      // Revert changes if connection test failed
      configManager.setApiUrl(apiUrl);
      apiClient.updateApiUrl(apiUrl);
      setTempApiUrl(apiUrl);
    } finally {
      setSavingApiUrl(false);
    }
  };

  const handleResetApiUrl = () => {
    const defaultUrl = configManager.getDefaultApiUrl();
    setTempApiUrl(defaultUrl);
    setApiUrlError(null);
  };

  const isApiUrlChanged = tempApiUrl !== apiUrl;

  const handleExportTrainingData = async () => {
    try {
      setExportingData(true);
      setTrainingError(null);
      setExportSuccess(null);
      
      const result = await trainingApi.exportData();
      const backendResult = result as unknown as Record<string, unknown>;
      setExportSuccess(`Training data exported successfully! ${backendResult.images_count} images, ${backendResult.labels_count} labels, ${result.total_annotations} annotations.`);
      
      // Clear success message after 10 seconds
      setTimeout(() => setExportSuccess(null), 10000);
      
      // Refresh training status
      const newTrainingStatus = await trainingApi.getStatus().catch(() => null);
      setTrainingStatus(newTrainingStatus);
      
    } catch (err) {
      setTrainingError(err instanceof Error ? err.message : 'Failed to export training data');
    } finally {
      setExportingData(false);
    }
  };

  const handleRetrainModel = async () => {
    try {
      setRetrainingModel(true);
      setTrainingError(null);
      setRetrainSuccess(null);
      
      const result = await trainingApi.retrain({
        custom_model_name: `retrained_model_${new Date().toISOString().split('T')[0]}`,
        description: `Retrained model with user feedback data from ${new Date().toLocaleDateString()}`
      });
      
      setRetrainSuccess(`Model retraining started! Job ID: ${result.retrain_job_id}. Estimated duration: ${result.estimated_duration_minutes || 'unknown'} minutes.`);
      
      // Clear success message after 15 seconds
      setTimeout(() => setRetrainSuccess(null), 15000);
      
      // Refresh training status
      const newTrainingStatus = await trainingApi.getStatus().catch(() => null);
      setTrainingStatus(newTrainingStatus);
      
    } catch (err) {
      setTrainingError(err instanceof Error ? err.message : 'Failed to start model retraining');
    } finally {
      setRetrainingModel(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getStatusIcon = (status: boolean | string) => {
    if (typeof status === 'boolean') {
      return status ? (
        <CheckCircle className="h-4 w-4 text-green-500" />
      ) : (
        <XCircle className="h-4 w-4 text-red-500" />
      );
    }
    
    if (status === 'healthy' || status === 'running') {
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    }
    
    return <AlertCircle className="h-4 w-4 text-yellow-500" />;
  };

  const getStatusBadge = (status: boolean | string) => {
    if (typeof status === 'boolean') {
      return (
        <Badge variant={status ? 'default' : 'destructive'}>
          {status ? 'Active' : 'Inactive'}
        </Badge>
      );
    }
    
    if (status === 'healthy' || status === 'running') {
      return <Badge variant="default">{status}</Badge>;
    }
    
    return <Badge variant="secondary">{status}</Badge>;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <div className="container mx-auto p-8">
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading settings...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-3">
            <Settings className="h-8 w-8 text-foreground" />
            <div>
              <h1 className="text-3xl font-bold text-foreground">Settings</h1>
              <p className="text-muted-foreground">System configuration and status</p>
            </div>
          </div>
          <div className="flex space-x-3">
            <Button onClick={fetchData} variant="outline">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button onClick={handleReloadConfig} disabled={reloading}>
              <Server className="h-4 w-4 mr-2" />
              {reloading ? 'Reloading...' : 'Reload Config'}
            </Button>
          </div>
        </div>

        {error && (
          <Card className="mb-6 border-destructive/50 bg-destructive/10">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-2 text-destructive">
                <XCircle className="h-5 w-5" />
                <span>{error}</span>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* API Configuration */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Globe className="h-5 w-5" />
                <span>API Configuration</span>
              </CardTitle>
              <CardDescription>Configure the API server URL for the frontend to connect to</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-2 space-y-2">
                  <label htmlFor="api-url" className="text-sm font-medium">API Server URL</label>
                  <Input
                    id="api-url"
                    type="url"
                    value={tempApiUrl}
                    onChange={(e) => setTempApiUrl(e.target.value)}
                    placeholder="http://localhost:8000"
                    className={apiUrlError ? "border-destructive" : ""}
                  />
                  {apiUrlError && (
                    <p className="text-sm text-destructive">{apiUrlError}</p>
                  )}
                  <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <span>Current:</span>
                    <code className="bg-muted px-2 py-1 rounded text-xs">{apiUrl}</code>
                    {configManager.isUsingCustomApiUrl() && (
                      <Badge variant="secondary" className="text-xs">Custom</Badge>
                    )}
                  </div>
                </div>
                <div className="flex flex-col justify-end space-y-2">
                  <Button 
                    onClick={handleSaveApiUrl} 
                    disabled={!isApiUrlChanged || savingApiUrl}
                    className="w-full"
                  >
                    <Save className="h-4 w-4 mr-2" />
                    {savingApiUrl ? 'Saving...' : 'Save & Test'}
                  </Button>
                  <Button 
                    onClick={handleResetApiUrl} 
                    variant="outline"
                    className="w-full"
                  >
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset to Default
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* System Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Monitor className="h-5 w-5" />
                <span>System Status</span>
              </CardTitle>
              <CardDescription>Current system health and runtime information</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {systemStatus ? (
                <>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Overall Status</span>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(systemStatus.status)}
                      {getStatusBadge(systemStatus.status)}
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Configuration Loaded</span>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(systemStatus.configuration_loaded)}
                      {getStatusBadge(systemStatus.configuration_loaded)}
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Database Connected</span>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(systemStatus.database_connected)}
                      {getStatusBadge(systemStatus.database_connected)}
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">ML Model Loaded</span>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(systemStatus.ml_model_loaded)}
                      {getStatusBadge(systemStatus.ml_model_loaded)}
                    </div>
                  </div>
                  <div className="pt-4 border-t">
                    <div className="text-sm">
                      <div>
                        <span className="text-muted-foreground">Version</span>
                        <p className="font-medium">{systemStatus.version}</p>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <p className="text-muted-foreground">Unable to load system status</p>
              )}
            </CardContent>
          </Card>

          {/* Health Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Cpu className="h-5 w-5" />
                <span>Health Check</span>
              </CardTitle>
              <CardDescription>API health and response status</CardDescription>
            </CardHeader>
            <CardContent>
              {healthStatus ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">API Status</span>
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(healthStatus.status)}
                      {getStatusBadge(healthStatus.status)}
                    </div>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Last checked: {new Date(healthStatus.timestamp).toLocaleString()}
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">Unable to load health status</p>
              )}
            </CardContent>
          </Card>

          {/* Camera Configuration */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Camera className="h-5 w-5" />
                <span>Camera Sources</span>
              </CardTitle>
              <CardDescription>Configured camera sources and their status</CardDescription>
            </CardHeader>
            <CardContent>
              {cameras ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between mb-4">
                    <div className="text-sm text-muted-foreground">
                      {cameras.enabled} of {cameras.total} cameras enabled
                    </div>
                    <Badge variant="outline">
                      {cameras.total} Total
                    </Badge>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {cameras.cameras.map((camera, index) => (
                      <Card key={index}>
                        <CardContent className="pt-4">
                          <div className="flex items-center justify-between mb-2">
                            <h4 className="font-medium">{camera.name}</h4>
                            {getStatusBadge(camera.enabled)}
                          </div>
                          <div className="space-y-2 text-sm text-muted-foreground">
                            <div>
                              <span className="font-medium">URL:</span> {camera.url}
                            </div>
                            <div>
                              <span className="font-medium">Interval:</span> {camera.interval_seconds}s
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">Unable to load camera configuration</p>
              )}
            </CardContent>
          </Card>

          {/* YOLO Configuration */}
          {systemConfig && (
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="h-5 w-5" />
                  <span>ML Model Configuration</span>
                </CardTitle>
                <CardDescription>Object detection model settings</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <span className="text-sm font-medium text-muted-foreground">Model</span>
                    <p className="text-lg font-medium">{systemConfig.global_.ml_model_config.model}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-muted-foreground">Confidence Threshold</span>
                    <p className="text-lg font-medium">{systemConfig.global_.ml_model_config.confidence_threshold}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-muted-foreground">Target Classes</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {systemConfig.global_.ml_model_config.target_classes.map((classId) => (
                        <Badge key={classId} variant="secondary" className="text-xs">
                          {classId}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="mt-6 pt-6 border-t">
                  <h4 className="font-medium mb-3">Global Settings</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Default Interval</span>
                      <p className="font-medium">{systemConfig.global_.default_interval_seconds}s</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Max Concurrent Fetches</span>
                      <p className="font-medium">{systemConfig.global_.max_concurrent_fetches}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Timeout</span>
                      <p className="font-medium">{systemConfig.global_.timeout_seconds}s</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Training & ML Management */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="h-5 w-5" />
                <span>Training & ML Management</span>
              </CardTitle>
              <CardDescription>Export training data and retrain models with user feedback</CardDescription>
            </CardHeader>
            <CardContent>
              {trainingError && (
                <div className="mb-4 p-3 bg-destructive/10 border border-destructive/50 rounded-md">
                  <div className="flex items-center space-x-2 text-destructive">
                    <XCircle className="h-4 w-4" />
                    <span className="text-sm">{trainingError}</span>
                  </div>
                </div>
              )}

              {exportSuccess && (
                <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-md dark:bg-green-950/20 dark:border-green-800">
                  <div className="flex items-center space-x-2 text-green-700 dark:text-green-400">
                    <CheckCircle className="h-4 w-4" />
                    <span className="text-sm">{exportSuccess}</span>
                  </div>
                </div>
              )}

              {retrainSuccess && (
                <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-md dark:bg-blue-950/20 dark:border-blue-800">
                  <div className="flex items-center space-x-2 text-blue-700 dark:text-blue-400">
                    <CheckCircle className="h-4 w-4" />
                    <span className="text-sm">{retrainSuccess}</span>
                  </div>
                </div>
              )}

              {trainingStatus ? (
                <div className="space-y-6">
                  {/* Training Statistics */}
                  <div>
                    <h4 className="font-medium mb-3">Training Data Statistics</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold text-foreground">{trainingStatus.stats?.total_feedback_entries ?? 0}</div>
                        <div className="text-sm text-muted-foreground">Feedback Entries</div>
                      </div>
                      <div className="text-center p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold text-foreground">{trainingStatus.stats?.total_annotations ?? 0}</div>
                        <div className="text-sm text-muted-foreground">Total Annotations</div>
                      </div>
                      <div className="text-center p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold text-foreground">{trainingStatus.stats?.named_cats ?? 0}</div>
                        <div className="text-sm text-muted-foreground">Named Cats</div>
                      </div>
                    </div>
                  </div>

                  {/* Current Training Job */}
                  {trainingStatus.current_job && (
                    <div>
                      <h4 className="font-medium mb-3">Current Training Job</h4>
                      <div className="p-4 bg-muted rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Job ID: {trainingStatus.current_job.job_id}</span>
                          <Badge variant={trainingStatus.current_job.status === 'running' ? 'default' : 'secondary'}>
                            {trainingStatus.current_job.status}
                          </Badge>
                        </div>
                        {trainingStatus.current_job.progress !== undefined && (
                          <div className="text-sm text-muted-foreground">
                            Progress: {trainingStatus.current_job.progress}%
                          </div>
                        )}
                        {trainingStatus.current_job.estimated_completion && (
                          <div className="text-sm text-muted-foreground">
                            Estimated completion: {new Date(trainingStatus.current_job.estimated_completion).toLocaleString()}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Available Models */}
                  {trainingStatus.available_models.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-3">Available Models</h4>
                      <div className="space-y-2">
                        {trainingStatus.available_models.map((model, index) => {
                          const backendStatus = trainingStatus as BackendTrainingStatus;
                          const isCurrent = model.name === (backendStatus.current_model?.split("/").pop() || model.name);
                          return (
                            <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                              <div>
                                <div className="font-medium">{model.name}</div>
                                <div className="text-sm text-muted-foreground">
                                  Created: {new Date(model.created).toLocaleDateString()}
                                </div>
                                {model.description && (
                                  <div className="text-sm text-muted-foreground">{model.description}</div>
                                )}
                              </div>
                              <div className="flex items-center space-x-2">
                                {isCurrent && (
                                  <Badge variant="default">Current</Badge>
                                )}
                                <Badge variant="outline" className="text-xs">
                                  {model.path}
                                </Badge>
                                {!isCurrent && (
                                  <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={async () => {
                                      setLoading(true);
                                      try {
                                        await trainingApi.switchModel(model.name);
                                        // Refresh status after switching
                                        const newStatus = await trainingApi.getStatus();
                                        setTrainingStatus(newStatus);
                                      } catch (err) {
                                        setError(err instanceof Error ? err.message : 'Failed to switch model');
                                      } finally {
                                        setLoading(false);
                                      }
                                    }}
                                    disabled={loading}
                                    title={`Switch to ${model.name}`}
                                  >
                                    Switch
                                  </Button>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Training Actions */}
                  <div>
                    <h4 className="font-medium mb-3">Training Actions</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="flex flex-col items-center p-6 bg-muted rounded-lg shadow">
                        <Button
                          onClick={handleExportTrainingData}
                          disabled={exportingData}
                          variant="outline"
                          className="w-full flex items-center justify-center space-x-2"
                          title="Export all feedback and annotations as a YOLO-format dataset for external training tools."
                        >
                          {exportingData ? (
                            <Loader2 className="h-5 w-5 animate-spin" />
                          ) : (
                            <Download className="h-5 w-5 text-green-600" />
                          )}
                          <span className="font-medium">Export Training Data</span>
                        </Button>
                        <span className="mt-2 text-xs text-muted-foreground text-center">
                          Generate YOLO format dataset from all labeled images.
                        </span>
                      </div>
                      <div className="flex flex-col items-center p-6 bg-muted rounded-lg shadow">
                        <Button
                          onClick={handleRetrainModel}
                          disabled={retrainingModel || (trainingStatus.stats?.total_annotations ?? 0) < 10}
                          className="w-full flex items-center justify-center space-x-2"
                          title="Retrain the current model using all feedback and annotations. Requires at least 10 annotations."
                        >
                          {retrainingModel ? (
                            <Loader2 className="h-5 w-5 animate-spin" />
                          ) : (
                            <Play className="h-5 w-5 text-blue-600" />
                          )}
                          <span className="font-medium">Retrain Model</span>
                        </Button>
                        <span className="mt-2 text-xs text-muted-foreground text-center">
                          {(trainingStatus.stats?.total_annotations ?? 0) < 10
                            ? 'Need at least 10 annotations'
                            : 'Fine-tune with feedback data'}
                        </span>
                      </div>
                    </div>
                    
                    <div className="mt-4 p-4 bg-muted rounded-lg">
                      <p className="text-sm text-muted-foreground">
                        <strong>Export Training Data:</strong> Generates a YOLO-format dataset with all user feedback and annotations for external training tools.
                      </p>
                      <p className="text-sm text-muted-foreground mt-2">
                        <strong>Retrain Model:</strong> Automatically fine-tunes the current model with user feedback to improve detection accuracy. 
                        Requires at least 10 annotations to start training.
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">Unable to load training status</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 