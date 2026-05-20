import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { startMietWebServer } from '../../packages/openclaw-miet-claw-plugin/runtime.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = __dirname;

await startMietWebServer({
  argv: process.argv.slice(2),
  projectRoot: path.resolve(webRoot, '..', '..'),
  webRoot,
  pythonBin: 'python3',
});
