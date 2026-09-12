# Hosting the QOALA web application

The build in `gui/dist` is static files. It needs no server-side anything: no
database, no functions, no accounts. Copy it somewhere that serves files and
it works.

## The one optional header pair

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

These turn on **cross-origin isolation**, which is what `SharedArrayBuffer`
and multi-core WebAssembly need. The application checks `crossOriginIsolated`
at startup and says which mode it is in. It works either way - isolation is a
speed-up, never a requirement - so a host that cannot set headers is fine.

Note that the shipped build runs the optimisation in an ordinary Web Worker,
which does **not** need these headers: the tab stays responsive on any host.
The headers are here so that a future threaded build has somewhere to look,
and so that anyone who wants them can set them.

## Pick your host

One file, whichever one your host reads:

| File | Host |
|---|---|
| `_headers` | Cloudflare Pages, Netlify |
| `vercel.json` | Vercel |
| `nginx.conf` | nginx - include the snippet in your `server` block |
| `.htaccess` | Apache with `mod_headers` |

Copy it into `gui/dist` before uploading (the GitHub Actions workflow does
this for Cloudflare Pages).

Nothing about the application depends on any of them. Moving host means
copying a different file.

## No host at all

```bash
cd gui/dist && python3 -m http.server 8000
```

Then open <http://localhost:8000>. That is the whole installation procedure.

## Keeping the address

Point a CNAME you control at whatever you deploy to, rather than publishing
the host's own address. Moving host is then a DNS change and nobody's
bookmarks break.
