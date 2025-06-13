/**
 * Color utilities for consistent cat coloring between frontend and backend
 */

// Same color palette as backend detection service
export const CAT_COLORS = [
  "#FF6B6B", // Red
  "#4ECDC4", // Teal
  "#45B7D1", // Blue
  "#96CEB4", // Green
  "#FFEAA7", // Yellow
  "#DDA0DD", // Plum
  "#98D8C8", // Mint
  "#F7DC6F", // Gold
  "#BB8FCE", // Light Purple
  "#85C1E9", // Light Blue
  "#F8C471", // Peach
  "#82E0AA", // Light Green
];

/**
 * Get a consistent color for a cat based on its name or index
 * @param catName - The cat's name (if available)
 * @param catIndex - The cat's index (0-based)
 * @returns Hex color string
 */
export function getCatColor(catName?: string, catIndex: number = 0): string {
  if (catName) {
    // Use hash of cat name for consistent color assignment (same logic as backend)
    const colorHash = Math.abs(hashString(catName)) % CAT_COLORS.length;
    return CAT_COLORS[colorHash];
  } else {
    // Fall back to index-based color for unnamed cats
    return CAT_COLORS[catIndex % CAT_COLORS.length];
  }
}

/**
 * Simple string hash function to match backend behavior
 * @param str - String to hash
 * @returns Hash number
 */
function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash;
}

/**
 * Get a lighter version of the cat color for backgrounds
 * @param catName - The cat's name (if available) 
 * @param catIndex - The cat's index (0-based)
 * @returns Hex color string with reduced opacity
 */
export function getCatColorLight(catName?: string, catIndex: number = 0): string {
  const color = getCatColor(catName, catIndex);
  return color + "20"; // Add 20% opacity
}

/**
 * Get the cat color as CSS custom property
 * @param catName - The cat's name (if available)
 * @param catIndex - The cat's index (0-based)  
 * @returns CSS color value
 */
export function getCatColorCSS(catName?: string, catIndex: number = 0): string {
  return getCatColor(catName, catIndex);
}