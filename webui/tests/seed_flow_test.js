/* Seed-mode behaviour without a browser DOM: models the page's state machine.
 * Run: node webui/tests/seed_flow_test.js
 */
"use strict";

const assert = require("assert");
const seed = require("../static/seed.js");

class FakeStorage {
  constructor() { this.map = new Map(); }
  getItem(key) { return this.map.has(key) ? this.map.get(key) : null; }
  setItem(key, value) { this.map.set(key, String(value)); }
  removeItem(key) { this.map.delete(key); }
}

class SeedPage {
  constructor(storage) {
    this.storage = storage;
    this.value = "";
    this.mode = "random";
    this.hints = "";
    this.error = "";
    this.restore();
  }
  restore() {
    const raw = this.storage.getItem("pika-h3-seed");
    if (!raw) { this.roll(); return; }
    const parsed = JSON.parse(raw);
    if (parsed.mode === "fixed") {
      const value = seed.parseSeed(parsed.value);
      this.value = (value === null ? seed.randomSeed(null) : value.toString());
      this.setMode("fixed");
    } else {
      this.roll();
    }
  }
  save() {
    this.storage.setItem("pika-h3-seed", JSON.stringify({ mode: this.mode, value: this.value }));
  }
  roll() {
    this.value = seed.randomSeed(null);
    this.setMode("random");
  }
  setMode(mode) {
    this.mode = mode === "fixed" ? "fixed" : "random";
    this.hints = this.mode === "fixed"
      ? "固定模式：当前种子会在每次提交时重复使用"
      : "随机模式：每次提交后自动换新的种子";
    this.save();
  }
  type(text) {
    this.value = String(text).replace(/[^0-9]/g, "").slice(0, 19);
    this.setMode("fixed");
  }
  plus() { this.value = seed.stepSeedText(this.value, 1); this.setMode("fixed"); }
  minus() { this.value = seed.stepSeedText(this.value, -1); this.setMode("fixed"); }
  submissionSeed() { return this.value; }
  afterSubmit() {
    if (this.mode === "random") this.roll();
    else this.save();
  }
}

// 1. Default is random with a concrete in-range value.
const storage = new FakeStorage();
const page = new SeedPage(storage);
assert.strictEqual(page.mode, "random");
assert.ok(seed.validateSeedText(page.value).valid);

// 2. Two consecutive submissions use different seeds and the display refreshes at POST time.
const first = page.submissionSeed();
page.afterSubmit();
const second = page.submissionSeed();
assert.notStrictEqual(first, second, "random submissions must not reuse a seed");
// 3. Manual typing switches to fixed and is not replaced.
page.type("42");
assert.strictEqual(page.mode, "fixed");
assert.strictEqual(page.submissionSeed(), "42");
page.afterSubmit();
assert.strictEqual(page.submissionSeed(), "42", "fixed seed must be reused");
page.afterSubmit();
assert.strictEqual(page.submissionSeed(), "42");

// 4. +1/-1 switch to fixed and clamp at 0.
page.type("5");
page.plus();
assert.strictEqual(page.value, "6");
assert.strictEqual(page.mode, "fixed");
page.type("0");
page.minus();
assert.strictEqual(page.value, "0");
page.type(seed.MAX_SEED.toString());
page.plus();
assert.strictEqual(page.value, seed.MAX_SEED.toString());

// 5. Mode persists across reloads; random regenerates on load, fixed restores.
page.type("987654321");
const reloadedFixed = new SeedPage(storage);
assert.strictEqual(reloadedFixed.mode, "fixed");
assert.strictEqual(reloadedFixed.value, "987654321");
reloadedFixed.roll();
assert.strictEqual(reloadedFixed.mode, "random");
const stored = JSON.parse(storage.getItem("pika-h3-seed"));
assert.strictEqual(stored.mode, "random");
const reloadedRandom = new SeedPage(storage);
assert.strictEqual(reloadedRandom.mode, "random");
assert.notStrictEqual(reloadedRandom.value, stored.value, "random mode draws a fresh value on load");

// 6. Invalid values are rejected.
assert.strictEqual(seed.validateSeedText("").valid, false);
assert.strictEqual(seed.validateSeedText("abc").valid, false);
assert.strictEqual(seed.validateSeedText("9223372036854775808").valid, false);
assert.strictEqual(seed.validateSeedText("9223372036854775807").valid, true);

// 7. 1000 draws stay in range and do not repeat consecutively.
let previous = null;
for (let index = 0; index < 1000; index += 1) {
  const value = seed.randomSeed(null);
  assert.ok(seed.validateSeedText(value).valid, value);
  assert.notStrictEqual(value, previous);
  previous = value;
}

console.log("seed flow: all assertions passed (random/fixed, refresh, persistence, validation)");
