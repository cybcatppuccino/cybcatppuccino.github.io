export const AI_LEVELS = Object.freeze([
  { level: 1, label: '1 · Learning', timeMs: 35, maxDepth: 2, multipv: 10, candidatePool: 10, noiseCp: 330, temperatureCp: 260, blunderRate: 0.58, endgameProbeMs: 0 },
  { level: 2, label: '2 · Beginner', timeMs: 55, maxDepth: 3, multipv: 9, candidatePool: 9, noiseCp: 270, temperatureCp: 220, blunderRate: 0.47, endgameProbeMs: 0 },
  { level: 3, label: '3 · Casual', timeMs: 85, maxDepth: 4, multipv: 8, candidatePool: 8, noiseCp: 210, temperatureCp: 175, blunderRate: 0.36, endgameProbeMs: 0 },
  { level: 4, label: '4 · Improving', timeMs: 135, maxDepth: 5, multipv: 7, candidatePool: 7, noiseCp: 155, temperatureCp: 130, blunderRate: 0.27, endgameProbeMs: 0 },
  { level: 5, label: '5 · Club', timeMs: 220, maxDepth: 6, multipv: 6, candidatePool: 6, noiseCp: 112, temperatureCp: 95, blunderRate: 0.19, endgameProbeMs: 0 },
  { level: 6, label: '6 · Strong club', timeMs: 340, maxDepth: 8, multipv: 5, candidatePool: 5, noiseCp: 78, temperatureCp: 68, blunderRate: 0.13, endgameProbeMs: 0 },
  { level: 7, label: '7 · Expert', timeMs: 520, maxDepth: 10, multipv: 5, candidatePool: 5, noiseCp: 52, temperatureCp: 45, blunderRate: 0.085, endgameProbeMs: 18 },
  { level: 8, label: '8 · Master', timeMs: 760, maxDepth: 12, multipv: 4, candidatePool: 4, noiseCp: 32, temperatureCp: 28, blunderRate: 0.05, endgameProbeMs: 30 },
  { level: 9, label: '9 · Elite', timeMs: 1150, maxDepth: 15, multipv: 3, candidatePool: 3, noiseCp: 16, temperatureCp: 13, blunderRate: 0.025, endgameProbeMs: 55 },
  { level: 10, label: '10 · Maximum', timeMs: 3600, maxDepth: 36, multipv: 1, candidatePool: 1, noiseCp: 0, temperatureCp: 0, blunderRate: 0, endgameProbeMs: 180 }
]);

export function levelConfig(level) {
  const numeric = Math.max(1, Math.min(10, Number(level || 5)));
  return AI_LEVELS[numeric - 1];
}

function gaussian(rng) {
  const u = Math.max(1e-9, rng());
  const v = Math.max(1e-9, rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Selects a legal searched line with a deliberately human-like error profile.
 * Scores are stored from White's perspective, so sideToMove determines utility.
 */
export function selectLineForLevel(lines, config, sideToMove = 'w', rng = Math.random) {
  if (!Array.isArray(lines) || !lines.length) return null;
  if (config.level >= 10 || lines.length === 1) return lines[0];
  const sign = sideToMove === 'b' ? -1 : 1;
  const candidates = lines.slice(0, Math.max(2, config.candidatePool || lines.length));
  const utilities = candidates.map(line => sign * Number(line.score || 0));

  // Do not turn a proven non-losing alternative into a forced mate blunder at
  // the upper levels. Beginners may still miss it; levels 7–9 receive a guard.
  let pool = candidates.map((line, index) => ({ line, index, utility: utilities[index] }));
  if (config.level >= 7) {
    const safe = pool.filter(item => item.utility > -28500);
    if (safe.length) pool = safe;
  }

  if (pool.length > 1 && rng() < config.blunderRate) {
    // Blunders are rank-biased rather than uniformly random: lower levels more
    // often choose a genuinely inferior searched move, but remain legal.
    const start = Math.min(pool.length - 1, config.level <= 3 ? 2 : 1);
    const tail = pool.slice(start);
    if (tail.length) {
      const exponent = 1.25 + (10 - config.level) * 0.12;
      const weights = tail.map((_, i) => Math.pow(i + 1, exponent));
      let pick = rng() * weights.reduce((sum, value) => sum + value, 0);
      for (let i = 0; i < tail.length; i += 1) {
        pick -= weights[i];
        if (pick <= 0) return tail[i].line;
      }
      return tail.at(-1).line;
    }
  }

  const noisy = pool.map(item => ({
    ...item,
    noisy: item.utility + gaussian(rng) * config.noiseCp
  }));
  const maxNoisy = Math.max(...noisy.map(item => item.noisy));
  const temperature = Math.max(1, config.temperatureCp || 1);
  const weights = noisy.map(item => Math.exp(Math.max(-12, (item.noisy - maxNoisy) / temperature)));
  let pick = rng() * weights.reduce((sum, value) => sum + value, 0);
  for (let i = 0; i < noisy.length; i += 1) {
    pick -= weights[i];
    if (pick <= 0) return noisy[i].line;
  }
  return noisy[0].line;
}
