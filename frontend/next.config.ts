import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for Docker
  output: 'standalone',
  
  // Enable experimental features for better development experience
  experimental: {
    // Enable faster refresh in development
    optimizePackageImports: ['lucide-react'],
  },
  
  // Configure images for better performance
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/api/**',
      },
    ],
  },
  
  // Enable source maps in development
  productionBrowserSourceMaps: false,
  
  // Configure webpack for better development experience
  webpack: (config, { dev, isServer }) => {
    if (dev && !isServer) {
      // Enable faster refresh in development
      config.watchOptions = {
        poll: 1000,
        aggregateTimeout: 300,
      };
    }
    return config;
  },

};

export default nextConfig;
