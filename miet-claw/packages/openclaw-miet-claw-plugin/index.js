import { definePluginEntry } from 'openclaw/plugin-sdk/plugin-entry';

import { MIET_PLUGIN_INFO, registerMietClawTools } from './runtime.js';

export default definePluginEntry({
  ...MIET_PLUGIN_INFO,
  register: registerMietClawTools,
});
