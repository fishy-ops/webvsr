"""Minimal Chrome DevTools Protocol driver, for capturing the real extension.

Store screenshots of a browser extension have to be of the extension actually
running, not a mockup -- so this drives the real Chrome that has the unpacked
extension loaded, and screenshots what it renders.
"""
import json, sys, time, urllib.request
import websocket


def targets(port=9222):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list") as r:
        return json.load(r)


def new_tab(url, port=9222):
    q = urllib.parse.quote(url, safe="")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/new?{q}") as r:
        return json.load(r)


class Tab:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.i = 0

    def send(self, method, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.i:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def goto(self, url, wait=3.0):
        self.send("Page.enable")
        self.send("Page.navigate", url=url)
        time.sleep(wait)

    def js(self, expr):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True,
                      awaitPromise=True)
        return r.get("result", {}).get("value")

    def shot(self, path, clip=None):
        kw = {"format": "png", "captureBeyondViewport": False}
        if clip:
            kw["clip"] = {**clip, "scale": 1}
        import base64
        data = self.send("Page.captureScreenshot", **kw)["data"]
        open(path, "wb").write(base64.b64decode(data))
        return path

    def metrics(self, w, h, dsf=2):
        self.send("Emulation.setDeviceMetricsOverride", width=w, height=h,
                  deviceScaleFactor=dsf, mobile=False)
