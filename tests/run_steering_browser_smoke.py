"""Exercise the real steering UI through a W3C WebDriver endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf"
DISCLOSURE = "100-prompt accepted pilot — exploratory, non-final"


class Driver:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        response = self.call(
            "POST",
            "/session",
            {
                "capabilities": {
                    "alwaysMatch": {
                        "browserName": "firefox",
                        "moz:firefoxOptions": {"args": ["-headless"]},
                    }
                }
            },
            session=False,
        )
        value = response["value"]
        self.session_id = value.get("sessionId") or response.get("sessionId")
        if not self.session_id:
            raise RuntimeError(f"WebDriver did not return a session ID: {response}")

    def call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        session: bool = True,
    ) -> dict[str, Any]:
        prefix = f"/session/{self.session_id}" if session else ""
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}{prefix}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"WebDriver {method} {path} failed: {detail}") from exc

    def find(self, selector: str) -> str:
        value = self.call(
            "POST", "/element", {"using": "css selector", "value": selector}
        )["value"]
        return value[ELEMENT_KEY]

    def find_all(self, selector: str) -> list[str]:
        values = self.call(
            "POST", "/elements", {"using": "css selector", "value": selector}
        )["value"]
        return [value[ELEMENT_KEY] for value in values]

    def click(self, selector: str) -> None:
        element = self.find(selector)
        self.call("POST", f"/element/{element}/click", {})

    def clear(self, selector: str) -> None:
        element = self.find(selector)
        self.call("POST", f"/element/{element}/clear", {})

    def type(self, selector: str, text: str) -> None:
        element = self.find(selector)
        self.call(
            "POST",
            f"/element/{element}/value",
            {"text": text, "value": list(text)},
        )

    def text(self, selector: str) -> str:
        element = self.find(selector)
        return self.call("GET", f"/element/{element}/text")["value"]

    def attribute(self, selector: str, name: str) -> str | None:
        element = self.find(selector)
        return self.call("GET", f"/element/{element}/attribute/{name}")["value"]

    def property(self, selector: str, name: str) -> Any:
        element = self.find(selector)
        return self.call("GET", f"/element/{element}/property/{name}")["value"]

    def execute(self, script: str) -> Any:
        return self.call("POST", "/execute/sync", {"script": script, "args": []})[
            "value"
        ]

    def wait(self, predicate: Any, *, timeout: float = 300.0) -> Any:
        deadline = time.monotonic() + timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                result = predicate()
                if result:
                    return result
            except (KeyError, RuntimeError) as exc:
                last_error = exc
            time.sleep(0.5)
        raise RuntimeError(f"browser wait timed out; last error: {last_error}")

    def screenshot(self, path: Path) -> None:
        encoded = self.call("GET", "/screenshot")["value"]
        path.write_bytes(base64.b64decode(encoded))

    def close(self) -> None:
        if getattr(self, "session_id", None):
            self.call("DELETE", "")
            self.session_id = ""


def _replace(driver: Driver, selector: str, value: str) -> None:
    driver.clear(selector)
    driver.type(selector, value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    desktop = output.with_name("nano-steering-browser-desktop.png")
    steer = output.with_name("nano-steering-browser-steer.png")
    swap = output.with_name("nano-steering-browser-swap.png")
    mobile = output.with_name("nano-steering-browser-mobile.png")
    mobile_inspector = output.with_name("nano-steering-browser-mobile-inspector.png")

    driver = Driver(args.webdriver)
    started = time.monotonic()
    try:
        driver.call("POST", "/window/rect", {"width": 1440, "height": 1000})
        driver.call("POST", "/url", {"url": args.url})
        driver.wait(lambda: driver.text("#serviceStatus") == "Model ready", timeout=60)
        if DISCLOSURE not in driver.text("body"):
            raise RuntimeError("pilot disclosure is not visible in the page")
        if driver.attribute("#chatTab", "aria-selected") != "true":
            raise RuntimeError("chat composer is not the default")
        if driver.property("#maxNewTokens", "value") != "32":
            raise RuntimeError("direct chat does not default to 32 output tokens")
        if driver.property("#readoutLayers", "value") != "13-50":
            raise RuntimeError("readouts do not default to inclusive layers 13-50")
        driver.wait(
            lambda: (
                "default 13-50"
                in str(
                    driver.execute(
                        "return document.getElementById('provenanceGrid').textContent;"
                    )
                )
            ),
            timeout=10,
        )

        _replace(
            driver,
            ".message-row textarea",
            "Write one short sentence about Spider-Man.",
        )
        _replace(driver, "#readoutLayers", "26-30")
        _replace(driver, "#topK", "4")
        _replace(driver, "#maxNewTokens", "16")
        driver.click("#generateButton")
        driver.wait(
            lambda: len(driver.find_all('button.top-token[data-action="source"]')) > 0
        )
        baseline_run = driver.text("#runIdentity")
        baseline_completion = driver.text('article[data-condition="clean"] .completion')
        if not baseline_completion.strip() or "<think>" in baseline_completion:
            raise RuntimeError(
                f"chat baseline is not a direct response: {baseline_completion!r}"
            )
        provenance_text = str(
            driver.execute(
                "return document.getElementById('provenanceGrid').textContent;"
            )
        ).lower()
        if "chat thinking" not in provenance_text or "disabled" not in provenance_text:
            raise RuntimeError("run provenance does not show non-thinking chat mode")

        desktop_layout = driver.execute(
            "const rect = (selector) => {"
            "  const value = document.querySelector(selector).getBoundingClientRect();"
            "  return {left: value.left, top: value.top, right: value.right, "
            "    bottom: value.bottom, width: value.width, height: value.height};"
            "};"
            "return {innerWidth: window.innerWidth, innerHeight: window.innerHeight, "
            "  bodyWidth: document.body.scrollWidth, workspace: rect('.workbench'), "
            "  run: rect('.run-pane'), inspector: rect('.inspector-pane'), "
            "  matrix: rect('.matrix-scroll')};"
        )
        if desktop_layout["bodyWidth"] > desktop_layout["innerWidth"] + 1:
            raise RuntimeError(f"desktop body overflows horizontally: {desktop_layout}")
        if desktop_layout["run"]["right"] > desktop_layout["inspector"]["left"] + 1:
            raise RuntimeError(f"desktop panes overlap: {desktop_layout}")
        for pane_name in ("run", "inspector"):
            pane = desktop_layout[pane_name]
            workspace = desktop_layout["workspace"]
            if (
                pane["width"] <= 0
                or pane["height"] <= 0
                or pane["left"] < workspace["left"] - 1
                or pane["right"] > workspace["right"] + 1
                or pane["top"] < workspace["top"] - 1
                or pane["bottom"] > workspace["bottom"] + 1
            ):
                raise RuntimeError(
                    f"desktop {pane_name} pane escapes the workbench: {desktop_layout}"
                )
        matrix_rect = desktop_layout["matrix"]
        inspector_rect = desktop_layout["inspector"]
        if (
            matrix_rect["width"] <= 0
            or matrix_rect["height"] <= 0
            or matrix_rect["left"] < inspector_rect["left"] - 1
            or matrix_rect["right"] > inspector_rect["right"] + 1
            or matrix_rect["top"] < inspector_rect["top"] - 1
            or matrix_rect["bottom"] > inspector_rect["bottom"] + 1
        ):
            raise RuntimeError(
                f"desktop matrix escapes the inspector: {desktop_layout}"
            )

        matrix_population = driver.execute(
            "const rows = document.querySelectorAll('#inspectorBody .matrix-row');"
            "const layerHeaders = document.querySelectorAll("
            "  '#inspectorBody .matrix-head > span[title^=\"Layer \"]'"
            ");"
            "const heatCells = document.querySelectorAll("
            "  '#inspectorBody .matrix-row .heat-cell');"
            "const populatedHeatCells = document.querySelectorAll("
            "  '#inspectorBody .matrix-row .heat-cell:not(.is-empty)');"
            "return {rows: rows.length, layerHeaders: layerHeaders.length, "
            "  heatCells: heatCells.length, populatedHeatCells: "
            "  populatedHeatCells.length, sourceButtons: document.querySelectorAll("
            "    '#inspectorBody button.top-token[data-action=\"source\"]'"
            "  ).length};"
        )
        if matrix_population["layerHeaders"] != 5:
            raise RuntimeError(
                f"inspector did not render five requested layers: {matrix_population}"
            )
        if matrix_population["rows"] < 4 or matrix_population["sourceButtons"] < 1:
            raise RuntimeError(
                f"inspector token matrix is not populated: {matrix_population}"
            )
        expected_cells = matrix_population["rows"] * matrix_population["layerHeaders"]
        if (
            matrix_population["heatCells"] != expected_cells
            or matrix_population["populatedHeatCells"] < 20
        ):
            raise RuntimeError(
                f"inspector heatmap matrix is incomplete: {matrix_population}"
            )
        aggregation = driver.execute(
            "const row = document.querySelector('#inspectorBody .matrix-row');"
            "const total = Number(row.querySelector('.count-cell').textContent);"
            "const layerCounts = [...row.querySelectorAll('.heat-cell')].map((cell) => {"
            "  const match = (cell.getAttribute('title') || '').match(/Layer \\d+; (\\d+) of/);"
            "  return match ? Number(match[1]) : 0;"
            "});"
            "row.dispatchEvent(new PointerEvent('pointerover', {bubbles: true}));"
            "return {total, layerCounts, layerSum: layerCounts.reduce((a, b) => a + b, 0), "
            "  scope: document.querySelector('select[data-action=\"position-scope\"]').value, "
            "  summary: document.querySelector('.position-scope-summary').textContent, "
            "  positionButtons: document.querySelectorAll('.position-token').length, "
            "  contributors: document.querySelectorAll('.position-token.is-contributor').length};"
        )
        if aggregation["scope"] != "all" or aggregation["positionButtons"] < 2:
            raise RuntimeError(f"readout positions do not default to all: {aggregation}")
        if aggregation["total"] != aggregation["layerSum"]:
            raise RuntimeError(f"row Count does not sum from Count by Layer: {aggregation}")
        if aggregation["contributors"] < 1:
            raise RuntimeError(f"row hover did not highlight contributing positions: {aggregation}")

        driver.execute(
            "document.querySelector('#inspectorBody .matrix-row').dispatchEvent("
            "  new PointerEvent('pointerout', {bubbles: true})"
            ");"
            "document.querySelectorAll('.position-token')[0].click();"
            "return true;"
        )
        driver.execute(
            "document.querySelectorAll('.position-token')[1].click(); return true;"
        )
        custom_scope = driver.execute(
            "return {"
            "  scope: document.querySelector('select[data-action=\"position-scope\"]').value,"
            "  summary: document.querySelector('.position-scope-summary').textContent,"
            "  selected: document.querySelectorAll('.position-token.is-selected').length,"
            "  maxCount: Math.max(...[...document.querySelectorAll('.count-cell')].map("
            "    (node) => Number(node.textContent)"
            "  ))"
            "};"
        )
        if (
            custom_scope["scope"] != "custom"
            or custom_scope["selected"] != 2
            or "2 positions selected" not in custom_scope["summary"]
            or custom_scope["maxCount"] > 10
        ):
            raise RuntimeError(f"custom multi-position scope is incorrect: {custom_scope}")
        driver.execute(
            "const select = document.querySelector('select[data-action=\"position-scope\"]');"
            "select.value = 'all';"
            "select.dispatchEvent(new Event('change', {bubbles: true}));"
            "return true;"
        )
        driver.wait(
            lambda: driver.property(
                'select[data-action="position-scope"]', "value"
            )
            == "all"
        )
        driver.screenshot(desktop)

        source_anchor = driver.execute(
            "const source = document.querySelector('button.top-token[data-action=\"source\"]');"
            "return {layer: source.dataset.layer, position: source.dataset.position, "
            "  tokenId: source.dataset.tokenId};"
        )
        driver.click('button.top-token[data-action="source"]')
        try:
            driver.wait(
                lambda: (
                    driver.execute(
                        "return document.activeElement && document.activeElement.id;"
                    )
                    == "closeInterventionButton"
                ),
                timeout=5,
            )
        except RuntimeError as exc:
            active_id = driver.execute(
                "return document.activeElement && document.activeElement.id;"
            )
            raise RuntimeError(
                "opening intervention did not focus its close control; "
                f"active element is {active_id!r}"
            ) from exc
        source_summary = driver.text("#sourceSummary")
        if (
            driver.property("#interventionLayers", "value")
            != source_anchor["layer"]
            or f"source layer {source_anchor['layer']}" not in source_summary
            or f"position {source_anchor['position']}" not in source_summary
        ):
            raise RuntimeError(
                "clicked aggregate row did not preserve its exact source anchor: "
                f"anchor={source_anchor!r}, summary={source_summary!r}"
            )
        _replace(driver, "#strengthNumber", "0.2")
        driver.click("#runInterventionButton")
        driver.wait(
            lambda: (
                driver.text("#runIdentity") != baseline_run
                and len(driver.find_all('article[data-condition="intervened"]')) == 1
            )
        )
        steer_run = driver.text("#runIdentity")
        steer_completion = driver.text(
            'article[data-condition="intervened"] .completion'
        )
        driver.screenshot(steer)

        driver.click('button[data-action="condition-tab"][data-condition="clean"]')
        driver.click('#inspectorBody button.top-token[data-action="source"]')
        selected_layer = driver.property("#interventionLayers", "value")
        _replace(driver, "#interventionLayers", "0-50")
        driver.click('input[name="interventionMode"][value="swap"]')
        if driver.property("#interventionLayers", "value") != "0-50":
            raise RuntimeError("mode change discarded the broad intervention range")
        _replace(driver, "#interventionLayers", selected_layer)
        _replace(driver, "#sourceTokenText", " Spider")
        driver.type("#targetTokenText", " fish")
        driver.wait(
            lambda: (
                "is-valid" in (driver.attribute("#sourceTokenization", "class") or "")
                and "is-valid"
                in (driver.attribute("#targetTokenization", "class") or "")
            )
        )
        source_value = driver.property("#sourceTokenText", "value")
        target_value = driver.property("#targetTokenText", "value")
        if source_value != " Spider":
            raise RuntimeError(f"swap source lost leading whitespace: {source_value!r}")
        if target_value != " fish":
            raise RuntimeError(f"swap target lost leading whitespace: {target_value!r}")
        driver.click("#runInterventionButton")
        driver.wait(lambda: driver.text("#runIdentity") != steer_run)
        swap_run = driver.text("#runIdentity")
        swap_completion = driver.text(
            'article[data-condition="intervened"] .completion'
        )
        driver.screenshot(swap)

        driver.call("POST", "/window/rect", {"width": 390, "height": 844})
        driver.execute("window.scrollTo(0, 0); return true;")
        time.sleep(1)
        layout = driver.execute(
            "return {innerWidth: window.innerWidth, bodyWidth: document.body.scrollWidth, "
            "header: document.querySelector('.app-header').getBoundingClientRect(), "
            "results: document.querySelector('.results-shell').getBoundingClientRect(), "
            "mobileDisclosure: {"
            "  text: document.querySelector('.mobile-pilot-disclosure').textContent, "
            "  display: getComputedStyle(document.querySelector("
            "    '.mobile-pilot-disclosure'"
            "  )).display"
            "}};"
        )
        driver.screenshot(mobile)
        if layout["bodyWidth"] > layout["innerWidth"] + 1:
            raise RuntimeError(f"mobile body overflows horizontally: {layout}")
        if (
            layout["mobileDisclosure"]["text"] != DISCLOSURE
            or layout["mobileDisclosure"]["display"] == "none"
        ):
            raise RuntimeError(f"mobile pilot disclosure is not visible: {layout}")

        driver.execute(
            "const pane = document.querySelector('.inspector-pane');"
            "const header = document.querySelector('.app-header');"
            "const top = pane.getBoundingClientRect().top + window.scrollY "
            "  - header.getBoundingClientRect().height - 7;"
            "window.scrollTo({top: Math.max(0, top), behavior: 'instant'});"
            "return true;"
        )
        time.sleep(1)
        mobile_inspector_layout = driver.execute(
            "const inspector = document.querySelector('.inspector-pane')"
            ".getBoundingClientRect();"
            "const matrix = document.querySelector('.matrix-scroll')"
            ".getBoundingClientRect();"
            "const header = document.querySelector('.app-header')"
            ".getBoundingClientRect();"
            "return {innerHeight: window.innerHeight, headerBottom: header.bottom, "
            "  inspectorTop: inspector.top, inspectorBottom: inspector.bottom, "
            "  matrixTop: matrix.top, matrixBottom: matrix.bottom};"
        )
        if (
            mobile_inspector_layout["inspectorBottom"]
            <= mobile_inspector_layout["headerBottom"]
            or mobile_inspector_layout["inspectorTop"]
            >= mobile_inspector_layout["innerHeight"]
            or mobile_inspector_layout["matrixBottom"]
            <= mobile_inspector_layout["headerBottom"]
            or mobile_inspector_layout["matrixTop"]
            >= mobile_inspector_layout["innerHeight"]
        ):
            raise RuntimeError(
                "mobile inspector did not scroll into the viewport: "
                f"{mobile_inspector_layout}"
            )
        driver.screenshot(mobile_inspector)

        return {
            "schema": "nemotron-steering-browser-smoke/v1",
            "disclosure": DISCLOSURE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": args.url,
            "browser": "Firefox via geckodriver",
            "baseline_run": baseline_run,
            "baseline_completion": baseline_completion,
            "steer_run": steer_run,
            "swap_run": swap_run,
            "steer_completion": steer_completion,
            "swap_completion": swap_completion,
            "source_exact_text": source_value,
            "target_exact_text": target_value,
            "desktop_layout": desktop_layout,
            "matrix_population": matrix_population,
            "aggregation": aggregation,
            "custom_scope": custom_scope,
            "source_anchor": source_anchor,
            "layout": layout,
            "mobile_inspector_layout": mobile_inspector_layout,
            "screenshots": {
                "desktop": str(desktop),
                "steer": str(steer),
                "swap": str(swap),
                "mobile": str(mobile),
                "mobile_inspector": str(mobile_inspector),
            },
            "elapsed_seconds": time.monotonic() - started,
            "passed": True,
        }
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webdriver", default="http://127.0.0.1:4444")
    parser.add_argument("--url", default="http://127.0.0.1:8000/")
    parser.add_argument("--output", default="artifacts/nano-steering-browser.json")
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
