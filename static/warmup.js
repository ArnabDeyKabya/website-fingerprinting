/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  /*
   * Implement this function to read n cache lines.
   * 1. Allocate a buffer of size n * LINESIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
   * 3. Collect total time taken in an array using `performance.now()`.
   * 4. Return the median of the time taken in milliseconds.
   */
  const buffer = new Uint8Array(n * LINESIZE);
  const iterations = 10;
  const times = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();

    for (let j = 0; j < n * LINESIZE; j += LINESIZE) {
      buffer[j]; // Access every LINESIZE-th byte
    }

    const end = performance.now();
    times.push(end - start);
  }

  // Sort and return median
  times.sort((a, b) => a - b);
  const mid = Math.floor(times.length / 2);
  return (times.length % 2 === 0)
    ? (times[mid - 1] + times[mid]) / 2
    : times[mid];
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
    const results = {};

    /* Call the readNlines function for n = 1, 10, ... 10,000,000 and store the result */
    const ns = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000];
    for (const n of ns) {
      try {
        const latency = readNlines(n);
        results[n] = latency;
      } catch (err) {
        console.error(`Error reading ${n} lines:`, err);
        break; // stop if memory limits are exceeded
      }
    }
    self.postMessage(results);
  }
});
