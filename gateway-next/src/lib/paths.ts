const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

/** Prefix static asset paths for GitHub Pages basePath exports. */
export function assetPath(path: string): string {
  if (!path.startsWith('/')) return path;
  return `${basePath}${path}`;
}
