const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v11.4.2</em>')) throw new Error('ries.html visible version should be v11.4.2');
if(!html.includes('ries-script.js?v=11.4.2')) throw new Error('ries-script cache tag should be v11.4.2');
const readme = fs.readFileSync('README.md','utf8');
if(/## RIES v\d/.test(readme) || /update|changelog/i.test(readme.replace('Detailed RIES release notes are kept under `changelog/`.',''))) throw new Error('README should not display release/update notes');
if(!fs.existsSync('changelog/RIES_v11.4.2_CHANGELOG.md')) throw new Error('v11.4.2 changelog missing from changelog folder');
const rootChangelogs = fs.readdirSync('.').filter(f => /^RIES_v.*_CHANGELOG\.md$/.test(f));
if(rootChangelogs.length) throw new Error('root changelog files should be moved into changelog/: '+rootChangelogs.join(','));
const context = {
  window: {}, document: {
    addEventListener: () => {}, body: {},
    getElementById: () => ({ addEventListener:()=>{}, disabled:false, hidden:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', setAttribute:()=>{}, appendChild:()=>{}, querySelectorAll:()=>[] }),
    querySelectorAll: () => [], createElement: () => ({ className:'', style:{}, prepend:()=>{} })
  }, console, performance: { now: () => 0 }, setTimeout, clearTimeout
};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/ries-harddb-v11_4_1-filtered.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);
const T = context.__RIES_CONSTDB_TEST__;
if(!T || typeof T.constantDbBudgetMs !== 'function') throw new Error('constant DB test hooks missing');
if(T.constantDbBudgetMs(4,16)!==20000) throw new Error('level 4 constant DB budget should be 20000');
if(T.constantDbBudgetMs(5,16)!==45000) throw new Error('level 5 constant DB budget should remain 45000');
if(T.constantDbBudgetMs(6,16)!==135000) throw new Error('level 6 constant DB budget should remain 135000');
console.log('PASS RIES v11.4.2 packaging and budget smoke test');
