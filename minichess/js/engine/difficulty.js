export const AI_LEVELS = Object.freeze([
  { level: 1, label: '1 · Beginner', timeMs: 90, maxDepth: 3, multipv: 5, margin: 260, temperature: 1.25 },
  { level: 2, label: '2 · Casual', timeMs: 140, maxDepth: 4, multipv: 5, margin: 210, temperature: 1.05 },
  { level: 3, label: '3 · Club novice', timeMs: 220, maxDepth: 5, multipv: 4, margin: 160, temperature: 0.85 },
  { level: 4, label: '4 · Club', timeMs: 340, maxDepth: 6, multipv: 4, margin: 120, temperature: 0.65 },
  { level: 5, label: '5 · Strong club', timeMs: 520, maxDepth: 8, multipv: 3, margin: 85, temperature: 0.45 },
  { level: 6, label: '6 · Expert', timeMs: 760, maxDepth: 10, multipv: 3, margin: 60, temperature: 0.30 },
  { level: 7, label: '7 · Master', timeMs: 1100, maxDepth: 13, multipv: 3, margin: 40, temperature: 0.20 },
  { level: 8, label: '8 · Elite', timeMs: 1600, maxDepth: 17, multipv: 2, margin: 24, temperature: 0.12 },
  { level: 9, label: '9 · Near maximum', timeMs: 2400, maxDepth: 23, multipv: 2, margin: 12, temperature: 0.05 },
  { level: 10, label: '10 · Maximum', timeMs: 3600, maxDepth: 36, multipv: 1, margin: 0, temperature: 0 }
]);

export function levelConfig(level) {
  const numeric = Math.max(1, Math.min(10, Number(level || 5)));
  return AI_LEVELS[numeric - 1];
}
