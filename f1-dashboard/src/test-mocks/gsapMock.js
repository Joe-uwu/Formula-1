// jest can't transform gsap's ESM plugin files (ScrollTrigger.js uses native
// `import`, and CRA's Jest config doesn't transform node_modules by default).
// These component tests exercise app logic, not animation timing, so gsap is
// mocked out entirely here rather than fighting Jest's transform pipeline.
const tween = { kill: () => {}, scrollTrigger: { kill: () => {} } };

export const gsap = {
  registerPlugin: () => {},
  set: () => tween,
  to: () => tween,
  fromTo: () => tween,
  ticker: { add: () => {}, remove: () => {}, lagSmoothing: () => {} },
};

export const ScrollTrigger = {
  create: () => ({ kill: () => {} }),
  update: () => {},
};

export default gsap;
