import { describe, expect, it } from "vitest";

import { SECTIONS } from "./sections";
import { REGIME_COLORS } from "./theme";

describe("theme", () => {
  it("tiene un color por cada categoría de régimen", () => {
    expect(Object.keys(REGIME_COLORS).sort()).toEqual(
      ["alcista", "bajista", "crash", "euforia", "neutral"],
    );
  });
});

describe("sections", () => {
  it("no repite ids", () => {
    const ids = SECTIONS.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
