import { configManager } from './config';

// Types based on the OpenAPI schema
export interface SystemConfig {
  global_: {
    default_interval_seconds: number;
    max_concurrent_fetches: number;
    timeout_seconds: number;
    ml_model_config: {
      confidence_threshold: number;
      model: string;
      target_classes: number[];
    };
  };
  images: Array<{
    enabled: boolean;
    interval_seconds: number;
    name: string;
    url: string;
  }>;
}

export interface SystemStatus {
  configuration_loaded: boolean;
  database_connected: boolean;
  enabled_cameras: number;
  status: string;
  total_cameras: number;
  version: string;
  ml_model_loaded: boolean;
}

export interface Camera {
  enabled: boolean;
  interval_seconds: number;
  name: string;
  url: string;
}

export interface CamerasResponse {
  cameras: Camera[];
  enabled: number;
  total: number;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
}

export interface DetectionImage {
  filename: string;
  source: string;
  timestamp: string;
  timestamp_display: string;
  file_size: number;
  file_size_mb: number;
  cat_count: number;
  max_confidence: number | null;
  activities_by_cat: Record<string, Array<{
    activity: string;
    confidence: number;
    reasoning: string;
    cat_index: number;
  }>>;
  has_feedback: boolean;
  has_detailed_annotations: boolean;
  inference_method: string;
  detections: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
    bounding_box: BoundingBox;
  }>;
  annotation_summary: string[];
}

export interface DetectionImagesResponse {
  images: DetectionImage[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  detection_path: string;
  has_feedback_data: boolean;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
}

export interface DetectionResult {
  detected: boolean;
  count: number;
  confidence: number;
  detections: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
    bounding_box: BoundingBox;
  }>;
}

export interface CameraFetchResult {
  message: string;
  results: Array<{
    camera: string;
    success: boolean;
    error?: string;
    detection_result?: DetectionResult;
  }>;
}

export interface SingleCameraFetchResult {
  message: string;
  result: {
    camera: string;
    success: boolean;
    error?: string;
    detection_result?: DetectionResult;
  };
}

export interface ImageAnnotations {
  filename: string;
  has_annotations: boolean;
  feedback_id?: string;
  original_detections?: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
    bounding_box: BoundingBox;
  }>;
  user_annotations?: Array<{
    cat_name?: string;
    correct_activity?: string;
    activity_feedback?: string;
    bounding_box: BoundingBox;
  }>;
  feedback_type?: string;
  notes?: string;
  timestamp?: string;
  user_id?: string;
}

export interface ReprocessResult {
  filename: string;
  source: string;
  reprocessed: boolean;
  previous_record_deleted: boolean;
  detection_result: {
    detected: boolean;
    count: number;
    confidence: number | null;
    detections: Array<{
      class_id: number;
      class_name: string;
      confidence: number;
      bounding_box: BoundingBox;
    }>;
    activities: Array<{
      activity: string;
      confidence: number;
      reasoning: string;
      cat_index: number;
    }>;
    activities_by_cat: Record<string, Array<{
      activity: string;
      confidence: number;
      reasoning: string;
      cat_index: number;
    }>>;
    primary_activity: string | null;
  };
  reprocess_timestamp: string;
}

export interface BulkReprocessResult {
  message: string;
  total_images: number;
  processed: number;
  errors: number;
  results: Array<{
    filename: string;
    source?: string;
    success: boolean;
    detected?: boolean;
    count?: number;
    confidence?: number;
    previous_record_deleted?: boolean;
    error?: string;
  }>;
  reprocess_timestamp: string;
}

// API Client class
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = configManager.getApiUrl() || 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  // Method to update the API URL
  public updateApiUrl(newUrl: string): void {
    this.baseUrl = newUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    // Always use the current API URL from configManager to support runtime changes
    const currentBaseUrl = configManager.getApiUrl();
    const url = `${currentBaseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // System endpoints
  async getSystemConfig(): Promise<SystemConfig> {
    return this.request<SystemConfig>('/system/config');
  }

  async getSystemStatus(): Promise<SystemStatus> {
    return this.request<SystemStatus>('/system/status');
  }

  async getHealthStatus(): Promise<HealthStatus> {
    return this.request<HealthStatus>('/system/health');
  }

  async reloadConfiguration(): Promise<{ message: string; enabled_images: number; total_images: number; ml_model: string }> {
    return this.request('/system/config/reload', {
      method: 'POST',
    });
  }

  async getRecentLogs(lines: number = 100): Promise<{ logs: string[]; total_lines: number }> {
    return this.request(`/system/logs?lines=${lines}`);
  }

  // Camera endpoints
  async getCameras(): Promise<CamerasResponse> {
    return this.request<CamerasResponse>('/cameras');
  }

  async fetchAllCameras(): Promise<CameraFetchResult> {
    return this.request('/cameras/fetch-all', {
      method: 'POST',
    });
  }

  async fetchCameraImage(cameraName: string): Promise<SingleCameraFetchResult> {
    return this.request(`/cameras/${encodeURIComponent(cameraName)}/fetch`, {
      method: 'POST',
    });
  }

  // Detection endpoints
  async getDetectionImages(page: number = 1, limit: number = 20): Promise<DetectionImagesResponse> {
    return this.request<DetectionImagesResponse>(`/detections/images?page=${page}&limit=${limit}`);
  }

  async getImageAnnotations(imageFilename: string): Promise<ImageAnnotations> {
    return this.request(`/detections/images/${encodeURIComponent(imageFilename)}/annotations`);
  }

  async reprocessImage(imageFilename: string): Promise<ReprocessResult> {
    return this.request(`/detections/images/${encodeURIComponent(imageFilename)}/reprocess`, {
      method: 'POST'
    });
  }

  async reprocessAllImages(): Promise<BulkReprocessResult> {
    return this.request('/detections/images/reprocess-all', {
      method: 'POST'
    });
  }

  // Feedback endpoints
  async submitFeedback(feedback: ImageFeedback): Promise<FeedbackSubmissionResult> {
    return this.request('/feedback', {
      method: 'POST',
      body: JSON.stringify(feedback),
    });
  }

  async getFeedbackList(): Promise<FeedbackListResponse> {
    return this.request<FeedbackListResponse>('/feedback');
  }

  async getFeedbackDetails(feedbackId: string): Promise<ImageFeedback> {
    return this.request(`/feedback/${encodeURIComponent(feedbackId)}`);
  }

  async deleteFeedback(feedbackId: string): Promise<{ message: string; deleted_feedback_id: string }> {
    return this.request(`/feedback/${encodeURIComponent(feedbackId)}`, {
      method: 'DELETE',
    });
  }

  // Training endpoints
  async exportTrainingData(): Promise<TrainingDataExportResult> {
    return this.request('/training/export', {
      method: 'POST',
    });
  }

  async retrainModel(request: ModelRetrainRequest = {}): Promise<ModelRetrainResult> {
    return this.request('/training/retrain', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getTrainingStatus(): Promise<TrainingStatus> {
    return this.request<TrainingStatus>('/training/status');
  }

  async switchModel(model: string): Promise<{ message: string; current_model: string }> {
    return this.request('/training/switch-model', {
      method: 'POST',
      body: JSON.stringify({ model }),
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// Export singleton instance
export const apiClient = new ApiClient();

// Export individual functions for easier use
export const systemApi = {
  getConfig: () => apiClient.getSystemConfig(),
  getStatus: () => apiClient.getSystemStatus(),
  getHealth: () => apiClient.getHealthStatus(),
  reloadConfig: () => apiClient.reloadConfiguration(),
  getLogs: (lines?: number) => apiClient.getRecentLogs(lines),
};

export const cameraApi = {
  list: () => apiClient.getCameras(),
  fetchAll: () => apiClient.fetchAllCameras(),
  fetchOne: (name: string) => apiClient.fetchCameraImage(name),
};

// Feedback types
export interface FeedbackAnnotation {
  class_id: number;
  class_name: string;
  bounding_box: BoundingBox;
  confidence?: number;
  cat_name?: string;
  activity_feedback?: string;
  correct_activity?: string;
  activity_confidence?: number;
}

export interface ImageFeedback {
  image_filename: string;
  image_path: string;
  original_detections: Array<{
    class_id: number;
    class_name: string;
    confidence: number;
    bounding_box: BoundingBox;
  }>;
  user_annotations: FeedbackAnnotation[];
  feedback_type: 'correction' | 'addition' | 'confirmation' | 'rejection';
  notes?: string;
  timestamp: string;
  user_id?: string;
}

export interface FeedbackSubmissionResult {
  message: string;
  feedback_id: string;
  image_filename: string;
  feedback_type: string;
  annotations_count: number;
  named_cats: string[];
  activity_feedback_count: number;
  timestamp: string;
  persisted: boolean;
}

export interface FeedbackListResponse {
  total_feedback: number;
  feedback: Array<{
    feedback_id: string;
    image_filename: string;
    feedback_type: string;
    annotations_count: number;
    timestamp: string;
    user_id: string;
    notes?: string;
  }>;
  persisted: boolean;
}

// Training interfaces
export interface TrainingDataExportResult {
  message: string;
  export_path: string;
  images_exported: number;
  labels_exported: number;
  total_annotations: number;
  export_timestamp: string;
  classes: Record<number, string>;
}

export interface ModelRetrainRequest {
  custom_model_name?: string;
  description?: string;
}

export interface ModelRetrainResult {
  message: string;
  retrain_job_id: string;
  model_name: string;
  status: string;
  estimated_duration_minutes?: number;
  training_data_stats: {
    total_images: number;
    total_annotations: number;
    classes: Record<number, string>;
  };
}

export interface TrainingStatus {
  status: string;
  available_models: Array<{
    name: string;
    path: string;
    created: string;
    description?: string;
    is_current: boolean;
  }>;
  current_job?: {
    job_id: string;
    status: string;
    progress?: number;
    estimated_completion?: string;
  };
  stats: {
    total_feedback_entries: number;
    total_annotations: number;
    named_cats: number;
  };
}

export const detectionApi = {
  getImages: (page?: number, limit?: number) => apiClient.getDetectionImages(page, limit),
  getImageAnnotations: (filename: string) => apiClient.getImageAnnotations(filename),
  reprocessImage: (filename: string) => apiClient.reprocessImage(filename),
  reprocessAllImages: () => apiClient.reprocessAllImages(),
};

export const feedbackApi = {
  submit: (feedback: ImageFeedback) => apiClient.submitFeedback(feedback),
  list: () => apiClient.getFeedbackList(),
  getDetails: (feedbackId: string) => apiClient.getFeedbackDetails(feedbackId),
  delete: (feedbackId: string) => apiClient.deleteFeedback(feedbackId),
};

export const trainingApi = {
  exportData: () => apiClient.exportTrainingData(),
  retrain: (request?: ModelRetrainRequest) => apiClient.retrainModel(request),
  getStatus: () => apiClient.getTrainingStatus(),
  switchModel: (model: string) => apiClient.switchModel(model),
}; 