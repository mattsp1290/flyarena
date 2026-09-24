import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir:'./tests/e2e', testMatch:'subpath.spec.ts', workers:1, forbidOnly:Boolean(process.env.CI),
  use:{baseURL:'http://127.0.0.1:4174', launchOptions:{args:['--host-resolver-rules=MAP flyarena.test 127.0.0.1','--no-proxy-server']}},
  webServer:{command:'npm run build -- --base /fly/ && node scripts/verify/serve-subpath.mjs',url:'http://127.0.0.1:4174/fly/',reuseExistingServer:false}
});
