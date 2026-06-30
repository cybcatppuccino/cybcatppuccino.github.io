const {spawnSync} = require('child_process');
const path = require('path');

const suites = [
  'tools/test_ries_packaging_startup.js',
  'tools/test_ries_latex_rendering.js',
  'tools/test_ries_database_modules.js',
  'tools/test_ries_constdb_lfunc_log.js',
  'tools/test_ries_precision_integer_sorting.js',
];

for(const suite of suites){
  const res = spawnSync(process.execPath, [suite], {stdio:'inherit', cwd:process.cwd(), env:process.env});
  if(res.status !== 0){
    console.error(`FAIL ${path.basename(suite)} exited with ${res.status}`);
    process.exit(res.status || 1);
  }
}
console.log(`PASS RIES consolidated test suite (${suites.length} files)`);
process.exit(0);
