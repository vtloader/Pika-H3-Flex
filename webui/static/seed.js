/* Seed arithmetic shared by the page and its Node test. No eval, no DOM. */
(() => {
  "use strict";
  const MAX_SEED = 9223372036854775807n;

  function randomSeed(cryptoObject) {
    const source = cryptoObject && typeof cryptoObject.getRandomValues === "function"
      ? cryptoObject : null;
    const buffer = new Uint8Array(8);
    if (source) {
      source.getRandomValues(buffer);
    } else {
      for (let index = 0; index < buffer.length; index += 1) {
        buffer[index] = Math.floor(Math.random() * 256);
      }
    }
    let value = 0n;
    for (const byte of buffer) value = (value << 8n) | BigInt(byte);
    return (value % (MAX_SEED + 1n)).toString();
  }

  function parseSeed(text) {
    const clean = String(text == null ? "" : text).replace(/[^0-9]/g, "");
    if (!clean) return null;
    try {
      return BigInt(clean);
    } catch (_) {
      return null;
    }
  }

  function clampSeed(value) {
    if (value < 0n) return 0n;
    if (value > MAX_SEED) return MAX_SEED;
    return value;
  }

  function stepSeedText(text, delta) {
    const current = parseSeed(text);
    const base = current === null ? 0n : current;
    return clampSeed(base + BigInt(delta)).toString();
  }

  function validateSeedText(text) {
    const value = parseSeed(text);
    if (value === null) return { valid: false, error: "种子必须是 0 到 2^63−1 之间的整数。" };
    if (value > MAX_SEED) return { valid: false, error: "种子必须小于 2^63。" };
    return { valid: true, value: value.toString(), error: "" };
  }

  const api = { MAX_SEED, randomSeed, parseSeed, clampSeed, stepSeedText, validateSeedText };
  if (typeof window !== "undefined") window.PikaSeed = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
