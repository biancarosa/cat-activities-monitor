'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
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
  Brain,
  Download,
  Play,
  BarChart3,
  Zap,
} from 'lucide-react';
import { systemApi, cameraApi, trainingApi, SystemConfig, SystemStatus, CamerasResponse, HealthStatus, TrainingStatus, ModelRetrainRequest, apiClient } from '@/lib/api';
import { configManager } from '@/lib/config';

export default function SettingsPage() {
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [cameras, setCameras] = useState<CamerasResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [reloading, setReloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // API URL configuration state
  const [apiUrl, setApiUrl] = useState(configManager.getApiUrl());
  const [tempApiUrl, setTempApiUrl] = useState(configManager.getApiUrl());
  const [savingApiUrl, setSavingApiUrl] = useState(false);
  const [apiUrlError, setApiUrlError] = useState<string | null>(null);

  // Training state
  const [exporting, setExporting] = useState(false);
  const [training, setTraining] = useState(false);
  const [switchingModel, setSwitchingModel] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [trainingConfig] = useState<ModelRetrainRequest>({
    train_yolo: true,
    train_cat_identification: true,
    include_clustering: true,
    parallel_training: false,
    yolo_config: {
      epochs: 50,
      batch_size: 16,
      base_model: 'yolo11l.pt'
    }
  });

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      setSuccessMessage(null);
      
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
      setTrainingStatus(trainingData);
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
      setExporting(true);
      setError(null);
      setSuccessMessage(null);
      const result = await trainingApi.exportData();
      setSuccessMessage(`Training data exported successfully! ${result.images_count} images with ${result.total_annotations} annotations.`);
      await fetchData(); // Refresh training status
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export training data');
    } finally {
      setExporting(false);
    }
  };

  const handleStartTraining = async () => {
    try {
      setTraining(true);
      setError(null);
      setSuccessMessage(null);
      const result = await trainingApi.retrain(trainingConfig);
      if (result.success) {
        const successfulTrainers = result.successful_trainers.join(', ');
        setSuccessMessage(`Training completed successfully! Trained: ${successfulTrainers}. Training time: ${result.total_training_time.toFixed(1)}s`);
      } else {
        setError(result.error_message || 'Training failed');
      }
      await fetchData(); // Refresh training status
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start training');
    } finally {
      setTraining(false);
    }
  };

  const handleSwitchModel = async (modelFilename: string) => {
    try {
      setSwitchingModel(true);
      setError(null);
      setSuccessMessage(null);
      const result = await trainingApi.switchModel(modelFilename);
      setSuccessMessage(`Model switched successfully to ${modelFilename}`);
      await fetchData(); // Refresh all data to show new current model
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to switch model');
    } finally {
      setSwitchingModel(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Auto-dismiss success messages after 5 seconds
  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

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

        {successMessage && (
          <Card className="mb-6 border-green-500/50 bg-green-500/10">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-2 text-green-600 dark:text-green-400">
                <CheckCircle className="h-5 w-5" />
                <span>{successMessage}</span>
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

          {/* AI Training Status */}
          {trainingStatus && (
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5" />
                  <span>AI Training Status</span>
                </CardTitle>
                <CardDescription>Machine learning model training and management</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Training Statistics */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-muted/30 rounded-lg">
                    <div className="text-2xl font-bold text-foreground">{trainingStatus.total_feedback}</div>
                    <div className="text-sm text-muted-foreground">Feedback Entries</div>
                  </div>
                  <div className="text-center p-4 bg-muted/30 rounded-lg">
                    <div className="text-2xl font-bold text-foreground">{trainingStatus.total_annotations}</div>
                    <div className="text-sm text-muted-foreground">Annotations</div>
                  </div>
                  <div className="text-center p-4 bg-muted/30 rounded-lg">
                    <div className="text-2xl font-bold text-foreground">{trainingStatus.unique_cats}</div>
                    <div className="text-sm text-muted-foreground">Unique Cats</div>
                  </div>
                  <div className="text-center p-4 bg-muted/30 rounded-lg">
                    <div className="text-2xl font-bold text-foreground">{trainingStatus.cat_profiles}</div>
                    <div className="text-sm text-muted-foreground">Cat Profiles</div>
                  </div>
                </div>

                {/* Training Readiness */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Overall Training</span>
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(trainingStatus.ready_for_training)}
                        {getStatusBadge(trainingStatus.ready_for_training ? 'Ready' : 'Not Ready')}
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">YOLO Training</span>
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(trainingStatus.yolo_training_ready)}
                        {getStatusBadge(trainingStatus.yolo_training_ready ? 'Ready' : 'Not Ready')}
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Cat ID Training</span>
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(trainingStatus.cat_id_training_ready)}
                        {getStatusBadge(trainingStatus.cat_id_training_ready ? 'Ready' : 'Not Ready')}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Training Actions */}
                <div className="flex flex-wrap gap-3 pt-4 border-t">
                  <Button 
                    onClick={handleExportTrainingData}
                    disabled={exporting || !trainingStatus.ready_for_training}
                    variant="outline"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    {exporting ? 'Exporting...' : 'Export Training Data'}
                  </Button>
                  <Button 
                    onClick={handleStartTraining}
                    disabled={training || !trainingStatus.ready_for_training}
                  >
                    <Play className="h-4 w-4 mr-2" />
                    {training ? 'Training...' : 'Start Training'}
                  </Button>
                </div>

                {/* Requirements */}
                {!trainingStatus.ready_for_training && (
                  <div className="mt-4 p-4 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                    <h4 className="font-medium text-orange-700 dark:text-orange-300 mb-2">Training Requirements</h4>
                    <div className="text-sm text-orange-600 dark:text-orange-400 space-y-1">
                      <div>• YOLO: ≥{trainingStatus.requirements.yolo_min_feedback} feedback, ≥{trainingStatus.requirements.yolo_min_annotations} annotations</div>
                      <div>• Cat ID: ≥{trainingStatus.requirements.cat_id_min_cats} cats, ≥{trainingStatus.requirements.cat_id_min_profiles} profiles, ≥{trainingStatus.requirements.cat_id_min_annotations} annotations</div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Available Models */}
          {trainingStatus && trainingStatus.available_models.length > 0 && (
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5" />
                  <span>Available Models</span>
                </CardTitle>
                <CardDescription>Manage and switch between trained models</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {trainingStatus.available_models.map((model, index) => (
                    <Card key={index} className={model.is_current ? 'border-primary' : ''}>
                      <CardContent className="pt-4">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium">{model.name}</h4>
                          <div className="flex items-center space-x-2">
                            {model.is_current && <Badge variant="default">Current</Badge>}
                            {model.is_custom && <Badge variant="secondary">Custom</Badge>}
                          </div>
                        </div>
                        <div className="space-y-2 text-sm text-muted-foreground">
                          <div>
                            <span className="font-medium">Size:</span> {model.size_mb === 'Download' ? 'Download' : `${model.size_mb} MB`}
                          </div>
                          {model.created && (
                            <div>
                              <span className="font-medium">Created:</span> {new Date(model.created).toLocaleDateString()}
                            </div>
                          )}
                          {model.description && (
                            <div>
                              <span className="font-medium">Description:</span> {model.description}
                            </div>
                          )}
                        </div>
                        {!model.is_current && model.size_mb !== 'Download' && (
                          <div className="mt-3">
                            <Button 
                              onClick={() => handleSwitchModel(model.filename)}
                              disabled={switchingModel}
                              size="sm"
                              variant="outline"
                              className="w-full"
                            >
                              <Zap className="h-4 w-4 mr-2" />
                              {switchingModel ? 'Switching...' : 'Switch to this Model'}
                            </Button>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
} 