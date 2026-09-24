// Test-only server: unlike Vite preview, never serves /data or /assets at root.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, sep, extname } from 'node:path';
const root = resolve('dist');
const types = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.json':'application/json', '.gz':'application/octet-stream' };
createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    if (!pathname.startsWith('/fly/')) { response.writeHead(404).end(); return; }
    const relative = pathname.slice(5) || 'index.html';
    const path = resolve(root, relative);
    if (!path.startsWith(root + sep)) { response.writeHead(404).end(); return; }
    const data = await readFile(path);
    response.writeHead(200, {'Content-Type':types[extname(path)] ?? 'application/octet-stream', 'Cache-Control':'no-store'});
    response.end(data);
  } catch { response.writeHead(404).end(); }
}).listen(4174, '127.0.0.1');
