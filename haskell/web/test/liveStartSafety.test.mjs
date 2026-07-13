import assert from "node:assert/strict";
import { test } from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { ComboLiveStartControls } from "../.tmp/web-tests/ComboLiveStartControls.js";

const combo = {
  id: 42,
  finalEquity: 1.25,
  openThreshold: 0.02,
  closeThreshold: 0.01,
  params: {
    binanceSymbol: "BTCUSDT",
    platform: "binance",
    interval: "1h",
    bars: 500,
    method: "10",
    normalization: "none",
    epochs: 5,
    hiddenSize: 16,
    learningRate: 0.001,
    valRatio: 0.2,
    patience: 2,
    slippage: 0.001,
    spread: 0.001,
  },
  source: "binance",
};

function buildProps(overrides = {}) {
  return {
    apiOk: "ok",
    topCombosLoading: false,
    topComboDisplay: combo,
    selectedCombo: combo,
    selectedComboStartLabel: "Start bot with combo #42",
    comboStartBlocked: false,
    comboStartBlockedReason: null,
    comboStartPending: false,
    refreshTopCombos: () => {},
    handleComboApply: () => {},
    handleComboStart: () => {},
    ...overrides,
  };
}

function visitElements(node, visit) {
  if (Array.isArray(node)) {
    for (const child of node) visitElements(child, visit);
    return;
  }
  if (!node || typeof node !== "object" || !("props" in node)) return;
  visit(node);
  visitElements(node.props.children, visit);
}

function buttonByText(props, text) {
  const tree = ComboLiveStartControls(props);
  let match = null;
  visitElements(tree, (element) => {
    if (element.type === "button" && element.props.children === text) match = element;
  });
  assert.ok(match, `expected button ${JSON.stringify(text)}`);
  return match;
}

test("applying a combo updates parameters without starting a live bot", () => {
  let applies = 0;
  let starts = 0;
  const props = buildProps({
    handleComboApply: (selected) => {
      assert.equal(selected, combo);
      applies += 1;
    },
    handleComboStart: () => {
      starts += 1;
    },
  });

  const apply = buttonByText(props, "Apply top combo now");
  apply.props.onClick();

  assert.equal(applies, 1);
  assert.equal(starts, 0);
});

test("a live start occurs only after the explicit start control is activated", () => {
  let starts = 0;
  const props = buildProps({ handleComboStart: () => (starts += 1) });

  const html = renderToStaticMarkup(ComboLiveStartControls(props));
  assert.match(html, />Start bot with combo #42<\/button>/);
  assert.equal(starts, 0, "rendering and receiving ready props must not start a bot");

  const start = buttonByText(props, "Start bot with combo #42");
  assert.equal(start.props.disabled, false);
  start.props.onClick();
  assert.equal(starts, 1);
});

test("stale, authentication, and error states fail closed and cannot trigger starts", () => {
  const blockedStates = [
    { label: "stale", apiOk: "unknown", comboStartBlocked: false, comboStartBlockedReason: null },
    { label: "authentication", apiOk: "auth", comboStartBlocked: false, comboStartBlockedReason: null },
    { label: "API error", apiOk: "down", comboStartBlocked: false, comboStartBlockedReason: "API request failed." },
    { label: "response error", apiOk: "ok", comboStartBlocked: true, comboStartBlockedReason: "Stale bot status response." },
  ];

  for (const state of blockedStates) {
    let starts = 0;
    const props = buildProps({ ...state, handleComboStart: () => (starts += 1) });
    const start = buttonByText(props, "Start bot with combo #42");

    assert.equal(start.props.disabled, true, `${state.label} state must disable live start`);
    assert.match(renderToStaticMarkup(ComboLiveStartControls(props)), /disabled=""/);
    start.props.onClick();
    assert.equal(starts, 0, `${state.label} state must guard even a stale/programmatic click`);
  }
});
