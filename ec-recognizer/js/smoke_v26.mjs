import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(process.cwd(), '..');
const appPath = path.join(root, 'app.js');
const cssPath = path.join(root, 'style.css');
const htmlPath = path.join(root, 'index.html');
const app = fs.readFileSync(appPath, 'utf8');
const css = fs.readFileSync(cssPath, 'utf8');
const html = fs.readFileSync(htmlPath, 'utf8');

function assert(cond, msg) { if (!cond) throw new Error(msg); }

assert(app.includes("const EC_ATLAS_VERSION = 'v26';"), 'runtime version should be v26');
assert(html.includes('>v26<'), 'HTML version badge should be v26');
assert(app.includes('visibleLevel: 7'), 'default visible level should be 1800');
assert(html.includes('<option value="7" selected>≈1800</option>'), '1800 option should be selected');
assert(html.includes('<span id="limit-value">≈1800</span>'), 'visible label should show 1800');
assert(app.includes('hoverStillMs: 500'), 'hover should be delayed by 500ms');
assert(app.includes('function scheduleHover'), 'stationary hover scheduler should exist');
assert(app.includes('Math.min(100, visibleCount)'), 'about top 100 detailed stars should be enabled');
assert(app.includes('function drawGridRing') && app.includes('projectedPolylineSegments'), 'v26 segmented grid path should exist');
assert(app.includes('clearHoverState(true);\n    markInteraction(260);'), 'pointerdown should clear hover before interaction');
assert(css.includes('translate(calc(var(--sel-size, 58px) * 0.38)'), 'selection ticks should orbit farther from center');
assert(css.includes('max-height:min(52vh,460px)'), 'search result area should be taller and scrollable');
assert(css.includes('.ciso-list{max-height:420px;}'), 'C-isogeny list should be taller');
assert(!app.includes('WebGLStarRenderer'), 'v26 must remain non-WebGL');
assert(!html.includes('atlas-gl'), 'v26 must not include WebGL canvas');

console.log('v26 smoke checks passed');
