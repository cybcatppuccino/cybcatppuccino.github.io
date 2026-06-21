import { readFileSync } from 'node:fs';
import { identifyCubicJS } from './ec_core.js';

const app = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
if (!app.includes("EC_ATLAS_VERSION = 'v24'")) throw new Error('version not updated to v24');
if (!app.includes('class WebGLStarRenderer')) throw new Error('WebGLStarRenderer missing');
if (!app.includes('state.glRenderer.render')) throw new Error('render path is not using WebGL');
if (app.includes('curves_by_j.json')) throw new Error('obsolete curves_by_j.json is still referenced');
if (app.includes('curves_compact.json')) throw new Error('obsolete curves_compact.json is still referenced');

const cubic = identifyCubicJS('u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0');
if (!cubic.ok || cubic.j !== '110592/233') throw new Error(`unexpected cubic j: ${JSON.stringify(cubic)}`);
console.log('v24 smoke ok');
