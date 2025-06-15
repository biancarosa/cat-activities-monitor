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
} from 'lucide-react';
import { systemApi, cameraApi, SystemConfig, SystemStatus, CamerasResponse, HealthStatus, apiClient } from '@/lib/api';
import { configManager } from '@/lib/config';

export default function SettingsPage() {
  const [systemConfig, setSystemConfig] = useState<SystemConfig | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [cameras, setCameras] = useState<CamerasResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [reloading, setReloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // API URL configuration state
  const [apiUrl, setApiUrl] = useState(configManager.getApiUrl());
  const [tempApiUrl, setTempApiUrl] = useState(configManager.getApiUrl());
  const [savingApiUrl, setSavingApiUrl] = useState(false);
  const [apiUrlError, setApiUrlError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [configData, statusData, camerasData, healthData] = await Promise.all([
        systemApi.getConfig().catch(() => null),
        systemApi.getStatus().catch(() => null),
        cameraApi.list().catch(() => null),
        systemApi.getHealth().catch(() => null),
      ]);

      setSystemConfig(configData);
      setSystemStatus(statusData);
      setCameras(camerasData);
      setHealthStatus(healthData);
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
        </div>
      </div>
    </div>
  );
} 