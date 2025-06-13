// Configuration management for the frontend application
const CONFIG_KEYS = {
  API_URL: 'cat-monitor-api-url',
} as const;

const DEFAULT_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class ConfigManager {
  private static instance: ConfigManager;
  private config: Map<string, string> = new Map();

  private constructor() {
    this.loadConfig();
  }

  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  private loadConfig(): void {
    // Only access localStorage on the client side
    if (typeof window !== 'undefined') {
      try {
        const savedApiUrl = localStorage.getItem(CONFIG_KEYS.API_URL);
        if (savedApiUrl) {
          this.config.set(CONFIG_KEYS.API_URL, savedApiUrl);
        } else {
          this.config.set(CONFIG_KEYS.API_URL, DEFAULT_API_URL);
        }
      } catch {
        console.warn('Failed to load config from localStorage');
        this.config.set(CONFIG_KEYS.API_URL, DEFAULT_API_URL);
      }
    } else {
      // Server-side fallback
      this.config.set(CONFIG_KEYS.API_URL, DEFAULT_API_URL);
    }
  }

  public getApiUrl(): string {
    return this.config.get(CONFIG_KEYS.API_URL) || DEFAULT_API_URL;
  }

  public setApiUrl(url: string): void {
    // Validate URL format
    try {
      new URL(url);
    } catch {
      throw new Error('Invalid URL format');
    }

    this.config.set(CONFIG_KEYS.API_URL, url);
    
    // Save to localStorage on client side
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem(CONFIG_KEYS.API_URL, url);
      } catch {
        console.warn('Failed to save config to localStorage');
      }
    }
  }

  public resetApiUrl(): void {
    this.config.set(CONFIG_KEYS.API_URL, DEFAULT_API_URL);
    
    // Remove from localStorage on client side
    if (typeof window !== 'undefined') {
      try {
        localStorage.removeItem(CONFIG_KEYS.API_URL);
      } catch {
        console.warn('Failed to remove config from localStorage');
      }
    }
  }

  public isUsingCustomApiUrl(): boolean {
    return this.getApiUrl() !== DEFAULT_API_URL;
  }

  public getDefaultApiUrl(): string {
    return DEFAULT_API_URL;
  }
}

// Export singleton instance
export const configManager = ConfigManager.getInstance(); 