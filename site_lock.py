"""The password page in front of the site's HTML pages (the project page, its gallery and the samples gallery).

    from site_lock import lock; path.write_text(lock(page_html))

GitHub Pages serves static files, so there is no login: each page's HTML is encrypted instead (AES-256-GCM, the key
derived from the password with PBKDF2-SHA256, ITERATIONS rounds) and replaced by a
small page that asks for the password and decrypts it in the browser (WebCrypto), then writes the decrypted page in its
place. The browser keeps the password (localStorage, one key for the whole github.io origin), so moving between the
pages asks only once. The salt and nonce come from an HMAC of the page under the password, so an unchanged page
encrypts to the same bytes on every build (a rebuild changes nothing in git) and a changed page gets a fresh salt, hence
a fresh key. This keeps the pages' text from casual visitors and search engines; the repos stay public, and
media files can still be fetched by anyone who has their URLs.

LOCKED below switches the whole thing on or off. The password is never written to a repo. It comes from the
SITE_PASSWORD environment variable or from PASSWORD_FILE (outside the repos). While LOCKED, a build without either stops, so a page cannot be published unlocked by accident;
SITE_PLAIN=1 writes the pages unencrypted, for a local check only.
"""
import base64
import hashlib
import hmac
import html
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# THE SWITCH. True: every page goes out behind the password page. False: the plain pages, with no password needed (for
# when the paper is out and the site may be public). Rebuild with build_site.py and publish both repos after changing it.
LOCKED = True

PASSWORD_FILE = Path.home() / ".config" / "aligning-video-motion" / "site_password"
ITERATIONS = 600_000
STORAGE_KEY = "aligning-video-motion:password"
# After the decrypted page is written in, scroll to the link's #anchor, as the browser would have on a plain page.
SCROLL = ('<script>if (location.hash) { const t = document.getElementById(decodeURIComponent(location.hash.slice(1)));'
          ' if (t) setTimeout(() => t.scrollIntoView(), 0); }</script>')


def password():
    if not LOCKED or os.environ.get("SITE_PLAIN") == "1":
        return None
    value = os.environ.get("SITE_PASSWORD") or (PASSWORD_FILE.read_text().strip() if PASSWORD_FILE.exists() else "")
    if not value:
        raise SystemExit(f"site_lock: no password (set SITE_PASSWORD or write {PASSWORD_FILE}); SITE_PLAIN=1 builds the "
                         "pages unlocked, for a local check only")
    return value


def lock(page, favicon="favicon.ico"):
    """The password page for `page` (a whole HTML document), or `page` itself under SITE_PLAIN=1."""
    secret = password()
    if secret is None:
        return page
    page = page.replace("</body>", SCROLL + "</body>", 1) if "</body>" in page else page + SCROLL
    digest = hmac.new(secret.encode(), page.encode(), hashlib.sha256).digest()
    salt, nonce = digest[:16], digest[16:28]   # deterministic per page and password (see the docstring)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(secret.encode())
    data = AESGCM(key).encrypt(nonce, page.encode(), None)
    b64 = lambda raw: base64.b64encode(raw).decode()  # noqa: E731
    payload = json.dumps(dict(salt=b64(salt), iv=b64(nonce), data=b64(data), iterations=ITERATIONS, key=STORAGE_KEY))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Password required</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<link rel="icon" href="{html.escape(favicon)}">
<!-- Encrypted by site_lock.py: the page decrypts in the browser with the right password. -->
<style>
body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #fff; color: #1f2328;
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
form {{ width: min(340px, 88vw); text-align: center; }}
h1 {{ font-size: 19px; font-weight: 600; margin: 0 0 16px; }}
input {{ width: 100%; box-sizing: border-box; padding: 10px 12px; font: inherit; border: 1px solid #c9ced6; border-radius: 6px; }}
input:focus {{ outline: 2px solid #2b6cb0; outline-offset: 1px; border-color: #2b6cb0; }}
button {{ width: 100%; margin-top: 10px; padding: 9px; font: inherit; font-weight: 600; color: #fff; background: #2b6cb0;
  border: 0; border-radius: 6px; cursor: pointer; }}
button:disabled {{ opacity: 0.6; cursor: default; }}
.error {{ min-height: 1.5em; margin: 10px 0 0; font-size: 14px; color: #b42318; }}
</style></head><body>
<form id="lock" hidden><h1>This page is password-protected</h1>
<input id="password" type="password" placeholder="Password" autocomplete="current-password" aria-label="Password" required>
<button type="submit">Open</button><p class="error" id="error" role="alert"></p></form>
<script>
(() => {{   // one scope: the decrypted page shares this window's globals, so nothing here may leak into them
  const locked = {payload};
  const bytes = (text) => Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
  async function decrypt(secret) {{
    const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), "PBKDF2", false, ["deriveKey"]);
    const key = await crypto.subtle.deriveKey(
      {{ name: "PBKDF2", salt: bytes(locked.salt), iterations: locked.iterations, hash: "SHA-256" }},
      base, {{ name: "AES-GCM", length: 256 }}, false, ["decrypt"]);
    return new TextDecoder().decode(await crypto.subtle.decrypt({{ name: "AES-GCM", iv: bytes(locked.iv) }}, key,
                                                                bytes(locked.data)));
  }}
  async function unlock(secret) {{   // false for a wrong password
    let page;
    try {{ page = await decrypt(secret); }} catch (error) {{ return false; }}
    try {{ localStorage.setItem(locked.key, secret); }} catch (error) {{ /* private mode: ask again next time */ }}
    document.open();
    document.write(page);
    document.close();
    return true;
  }}
  (async () => {{
    let saved = null;
    try {{ saved = localStorage.getItem(locked.key); }} catch (error) {{ /* storage blocked */ }}
    if (saved && await unlock(saved)) return;
    const form = document.getElementById("lock"), field = document.getElementById("password");
    form.hidden = false;
    field.focus();
    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const button = form.querySelector("button");
      button.disabled = true;
      if (!(await unlock(field.value))) {{
        document.getElementById("error").textContent = "That password isn't right.";
        button.disabled = false;
        field.select();
      }}
    }});
  }})();
}})();
</script>
<noscript><p>This page needs JavaScript to open.</p></noscript>
</body></html>
'''
