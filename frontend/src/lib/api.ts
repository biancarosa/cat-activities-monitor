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
    cat_uuid?: string;
    cat_name?: string;
    identification_suggestion?: {
      suggested_profile?: {
        uuid: string;
        name: string;
        description?: string;
      };
      confidence: number;
      is_confident_match: boolean;
      is_new_cat: boolean;
    };
    // Activity detection fields
    activity?: string;
    activity_confidence?: number;
    nearby_objects?: Array<{
      object_class: string;
      confidence: number;
      distance: number;
      relationship: string;
      interaction_type: string;
    }>;
    contextual_activity?: string;
    interaction_confidence?: number;
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
  detection_imgs_path: string;
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

  async retrainModel(request: ModelRetrainRequest = {}): Promise<ModelRetrainResult> {
    return this.request('/training/retrain', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getTrainingStatus(): Promise<TrainingStatus> {
    return this.request<TrainingStatus>('/training/status');
  }

  async suggestCatIdentifications(features: number[][]): Promise<CatIdentificationSuggestion[]> {
    return this.request('/training/cat-identification/suggest', {
      method: 'POST',
      body: JSON.stringify({ features }),
    });
  }

  async switchModel(model: string): Promise<{ message: string; current_model: string }> {
    return this.request('/training/switch-model', {
      method: 'POST',
      body: JSON.stringify({ model }),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  // Cat Profile public methods
  public async listCatProfiles(): Promise<CatProfileListResponse> {
    return this.request<CatProfileListResponse>('/cats');
  }
  public async createCatProfile(profileData: CreateCatProfileRequest): Promise<CatProfileResponse> {
    return this.request('/cats', {
      method: 'POST',
      body: JSON.stringify(profileData),
    });
  }
  public async getCatProfileByUuid(catUuid: string): Promise<CatProfile> {
    return this.request<CatProfile>(`/cats/by-uuid/${encodeURIComponent(catUuid)}`);
  }
  public async getCatProfileByName(catName: string): Promise<CatProfile> {
    return this.request<CatProfile>(`/cats/by-name/${encodeURIComponent(catName)}`);
  }
  public async updateCatProfile(catUuid: string, profileData: UpdateCatProfileRequest): Promise<CatProfileResponse> {
    return this.request(`/cats/${encodeURIComponent(catUuid)}`, {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  }
  public async deleteCatProfile(catUuid: string): Promise<{ message: string; deleted_cat_uuid: string; deleted_cat_name: string }> {
    return this.request(`/cats/${encodeURIComponent(catUuid)}`, {
      method: 'DELETE',
    });
  }
  public async getCatActivityHistory(catName: string): Promise<{
    cat_name: string;
    total_feedback_entries: number;
    activity_history: Array<{
      feedback_id: string;
      timestamp: string;
      image_filename: string;
      detected_activity?: string;
      activity_feedback?: string;
      confidence?: number;
      bounding_box: BoundingBox;
    }>;
    data_source: string;
  }> {
    return this.request(`/cats/by-name/${encodeURIComponent(catName)}/activity-history`);
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
  cat_profile_uuid?: string;
  corrected_class_id?: number;
  corrected_class_name?: string;
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
  images_count: number;
  labels_count: number;
  total_annotations: number;
  export_timestamp: string;
}

export interface ModelRetrainRequest {
  train_yolo?: boolean;
  train_cat_identification?: boolean;
  include_clustering?: boolean;
  parallel_training?: boolean;
  yolo_config?: {
    epochs?: number;
    batch_size?: number;
    base_model?: string;
  };
}

export interface ModelRetrainResult {
  success: boolean;
  total_training_time: number;
  successful_trainers: string[];
  training_results: Record<string, {
    success: boolean;
    model_path?: string;
    metrics?: Record<string, unknown>;
    training_time?: number;
    error_message?: string;
  }>;
  training_data_stats: {
    total_samples: number;
    feature_samples: number;
    unique_labels: number;
  };
  error_message?: string;
}

export interface TrainingStatus {
  ready_for_training: boolean;
  yolo_training_ready: boolean;
  cat_id_training_ready: boolean;
  total_feedback: number;
  total_annotations: number;
  unique_cats: number;
  cat_profiles: number;
  available_models: Array<{
    name: string;
    filename: string;
    model_name: string;
    size_mb: number | string;
    created: string | null;
    is_custom: boolean;
    is_current?: boolean;
    description?: string;
    metadata?: Record<string, unknown>;
  }>;
  requirements: {
    yolo_min_feedback: number;
    yolo_min_annotations: number;
    cat_id_min_cats: number;
    cat_id_min_profiles: number;
    cat_id_min_annotations: number;
  };
}

export interface CatIdentificationSuggestion {
  detection_index: number;
  suggested_profile?: {
    uuid: string;
    name: string;
    description?: string;
  };
  confidence: number;
  is_confident_match: boolean;
  is_new_cat: boolean;
  similarity_threshold: number;
  suggestion_threshold: number;
  top_matches: Array<{
    profile: {
      uuid: string;
      name: string;
      description?: string;
    };
    similarity: number;
  }>;
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
  retrain: (request?: ModelRetrainRequest) => apiClient.retrainModel(request),
  getStatus: () => apiClient.getTrainingStatus(),
  switchModel: (model: string) => apiClient.switchModel(model),
  suggestCatIdentifications: (features: number[][]) => apiClient.suggestCatIdentifications(features),
};

// Cat Profile interfaces
export interface CatProfile {
  cat_uuid: string;
  name: string;
  description?: string;
  color?: string;
  breed?: string;
  favorite_activities?: string[];
  created_timestamp: string;
  last_seen_timestamp?: string;
  total_detections: number;
  average_confidence: number;
  preferred_locations?: string[];
}

export interface CreateCatProfileRequest {
  name: string;
  description?: string;
  color?: string;
  breed?: string;
  favorite_activities?: string[];
}

export interface UpdateCatProfileRequest {
  name?: string;
  description?: string;
  color?: string;
  breed?: string;
  favorite_activities?: string[];
}

export interface CatProfileListResponse {
  total_cats: number;
  cats: CatProfile[];
  data_source: string;
}

export interface CatProfileResponse {
  message: string;
  cat_uuid: string;
  cat_name: string;
  created_timestamp?: string;
  persisted: boolean;
}

// Cat Profile API methods
export const catProfileApi = {
  list: (): Promise<CatProfileListResponse> => apiClient.listCatProfiles(),
  
  create: (profileData: CreateCatProfileRequest): Promise<CatProfileResponse> => 
    apiClient.createCatProfile(profileData),
  
  getByUuid: (catUuid: string): Promise<CatProfile> => 
    apiClient.getCatProfileByUuid(catUuid),
  
  getByName: (catName: string): Promise<CatProfile> => 
    apiClient.getCatProfileByName(catName),
  
  update: (catUuid: string, profileData: UpdateCatProfileRequest): Promise<CatProfileResponse> => 
    apiClient.updateCatProfile(catUuid, profileData),
  
  delete: (catUuid: string): Promise<{ message: string; deleted_cat_uuid: string; deleted_cat_name: string }> => 
    apiClient.deleteCatProfile(catUuid),
  
  getActivityHistory: (catName: string): Promise<{
    cat_name: string;
    total_feedback_entries: number;
    activity_history: Array<{
      feedback_id: string;
      timestamp: string;
      image_filename: string;
      detected_activity?: string;
      activity_feedback?: string;
      confidence?: number;
      bounding_box: BoundingBox;
    }>;
    data_source: string;
  }> => apiClient.getCatActivityHistory(catName),
};

// Dashboard interfaces
export interface DashboardOverview {
  time_period_hours: number;
  total_named_cats: number;
  named_cats_seen_recently: number;
  total_detections: number;
  total_cats_detected: number;
  top_locations: Array<{
    location: string;
    activity_count: number;
  }>;
  recent_activities: Array<{
    timestamp: string;
    location: string;
    cat_name?: string;
    activity?: string;
    confidence: number;
  }>;
  summary: {
    avg_cats_per_detection: number;
    most_active_location?: string;
    named_cats_list: string[];
  };
}

export interface CatActivitySummary {
  time_period_hours: number;
  cats: Array<{
    cat_name: string;
    cat_uuid: string;
    description?: string;
    total_detections: number;
    favorite_locations: Array<{
      location: string;
      count: number;
    }>;
    common_activities: Array<{
      activity: string;
      count: number;
    }>;
    last_seen?: string;
    last_location?: string;
    avg_confidence: number;
    recent_timeline: Array<{
      timestamp: string;
      location: string;
      activity: string;
      confidence: number;
    }>;
    is_active: boolean;
  }>;
  total_cats: number;
  active_cats: number;
}

export interface LocationActivitySummary {
  time_period_hours: number;
  locations: Array<{
    location: string;
    total_detections: number;
    total_cats_detected: number;
    unique_cats_count: number;
    unique_cats: string[];
    common_activities: Array<{
      activity: string;
      count: number;
    }>;
    hourly_activity_pattern: number[];
    peak_hour: number;
    avg_confidence: number;
    avg_cats_per_detection: number;
    recent_timeline: Array<{
      timestamp: string;
      cat_name?: string;
      activity: string;
      confidence: number;
    }>;
    camera_config?: {
      enabled: boolean;
      interval_seconds: number;
      url: string;
    };
  }>;
  total_locations: number;
}

export interface ActivityTimeline {
  time_period_hours: number;
  granularity: string;
  bucket_size_minutes: number;
  timeline: Array<{
    timestamp: string;
    total_detections: number;
    total_cats: number;
    locations: Record<string, number>;
    activities: Record<string, number>;
    named_cats: Record<string, number>;
    cat_activities: Record<string, Record<string, number>>;  // cat_name -> activity -> count
    unique_cats_count: number;
    most_active_location?: string;
    primary_activity?: string;
  }>;
  total_buckets: number;
  summary: {
    total_detections: number;
    total_cats_detected: number;
    peak_activity_time?: string;
  };
}

// Dashboard API methods
export const dashboardApi = {
  getOverview: async (hours: number = 24): Promise<DashboardOverview> => {
    const baseURL = configManager.getApiUrl();
    const response = await fetch(`${baseURL}/dashboard/overview?hours=${hours}`, {
      headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) throw new Error(`Failed to fetch dashboard overview: ${response.statusText}`);
    return response.json();
  },
  
  getCats: async (hours: number = 24): Promise<CatActivitySummary> => {
    const baseURL = configManager.getApiUrl();
    const response = await fetch(`${baseURL}/dashboard/cats?hours=${hours}`, {
      headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) throw new Error(`Failed to fetch cats dashboard: ${response.statusText}`);
    return response.json();
  },
  
  getLocations: async (hours: number = 24): Promise<LocationActivitySummary> => {
    const baseURL = configManager.getApiUrl();
    const response = await fetch(`${baseURL}/dashboard/locations?hours=${hours}`, {
      headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) throw new Error(`Failed to fetch locations dashboard: ${response.statusText}`);
    return response.json();
  },
  
  getTimeline: async (hours: number = 24, granularity: string = 'hour'): Promise<ActivityTimeline> => {
    const baseURL = configManager.getApiUrl();
    const response = await fetch(`${baseURL}/dashboard/timeline?hours=${hours}&granularity=${granularity}`, {
      headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) throw new Error(`Failed to fetch timeline dashboard: ${response.statusText}`);
    return response.json();
  },
}; 