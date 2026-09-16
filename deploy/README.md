# Hosting the QOALA web application

The application is already hosted at <https://rusty-qoala.pages.dev>, on
Cloudflare Pages. `.github/workflows/deploy.yml` builds and publishes it on
every push to `main`. This page is for hosting a copy of your own.

The build in `gui/dist` is static files. It needs no server-side anything: no
database, no functions, no accounts. Copy it somewhere that serves files and
it works.

## The one optional header pair

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

These turn on **cross-origin isolation**, which is what `SharedArrayBuffer`
and multi-core WebAssembly need.

**They do not make the shipped build faster.** The web application runs on
one core whether the page is cross-origin isolated or not: its WebAssembly is
built without threads, and the optimisation runs in an ordinary Web Worker,
which needs no special headers and keeps the tab responsive on any host. A
host that cannot set them loses nothing.

Multi-core runs are native only. The desktop application
(`cargo run -p qoala-gui --release`) and the library on the command line
spread ESCALADE over every core; QOALA runs on one core everywhere.

The headers are here so that a future threaded web build has somewhere to
look, and so that anyone who wants them can set them. `require-corp` blocks
cross-origin resources that do not opt in; the application loads none.

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
