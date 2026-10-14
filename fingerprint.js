// Shazam-style audio fingerprinting (Wang 2003): spectrogram -> constellation of peaks -> pair hashes -> offset voting.
// The same file runs in Node (building the song library) and in the browser (the mic), so both sides hash identically.
(function (root) {
  const SR = 11025;            // everything is analysed at 11,025 Hz mono
  const N = 1024, HOP = 256;   // 93 ms window, 23 ms hop (43 frames a second)
  const F_LO = 10, F_HI = 420; // bins kept: ~110 Hz .. 4.5 kHz, the range phone mics and speakers reproduce well
  const RF = 10, RT = 7;       // a peak must be the loudest point within +-10 bins and +-7 frames
  const PEAKS_PER_SEC = 18;    // strongest peaks kept per second (the "stars")
  const FANOUT = 6;            // each star pairs with this many later stars
  const DT_MIN = 1, DT_MAX = 63, DF_MAX = 120;

  const hann = new Float32Array(N);
  for (let i = 0; i < N; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);
  const LOG2N = Math.log2(N), rev = new Uint32Array(N);
  for (let i = 0; i < N; i++) { let r = 0; for (let b = 0; b < LOG2N; b++) r |= ((i >> b) & 1) << (LOG2N - 1 - b); rev[i] = r; }
  const cosT = new Float32Array(N / 2), sinT = new Float32Array(N / 2);
  for (let i = 0; i < N / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / N); sinT[i] = -Math.sin(2 * Math.PI * i / N); }
  const re = new Float32Array(N), im = new Float32Array(N);
  function fftMag(frame, out) {          // in-place radix-2 FFT, writes log magnitude of bins F_LO..F_HI
    for (let i = 0; i < N; i++) { re[rev[i]] = frame[i] * hann[i]; im[rev[i]] = 0; }
    for (let size = 2; size <= N; size <<= 1) {
      const half = size >> 1, step = N / size;
      for (let s = 0; s < N; s += size) for (let k = 0; k < half; k++) {
        const c = cosT[k * step], si = sinT[k * step], a = s + k, b = a + half;
        const tr = re[b] * c - im[b] * si, ti = re[b] * si + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
    for (let f = F_LO; f <= F_HI; f++) out[f - F_LO] = Math.log(re[f] * re[f] + im[f] * im[f] + 1e-10);
  }

  // log-magnitude spectrogram: rows = frames, each a Float32Array of NF bins
  const NF = F_HI - F_LO + 1;
  function spectrogram(pcm) {
    const nFrames = Math.max(0, Math.floor((pcm.length - N) / HOP) + 1), S = new Array(nFrames), fr = new Float32Array(N);
    for (let t = 0; t < nFrames; t++) { fr.set(pcm.subarray(t * HOP, t * HOP + N)); S[t] = new Float32Array(NF); fftMag(fr, S[t]); }
    return S;
  }

  // constellation: local maxima of the spectrogram, thinned to the strongest PEAKS_PER_SEC per second
  function peaks(S) {
    const T = S.length; if (!T) return [];
    const fmax = S.map(row => {                       // max over +-RF bins
      const o = new Float32Array(NF);
      for (let f = 0; f < NF; f++) { let m = -Infinity; for (let g = Math.max(0, f - RF); g <= Math.min(NF - 1, f + RF); g++) if (row[g] > m) m = row[g]; o[f] = m; }
      return o;
    });
    const cand = [];
    for (let t = 0; t < T; t++) {
      // whiten each frame by its median-ish level so quiet passages still produce stars
      let mean = 0; for (let f = 0; f < NF; f++) mean += S[t][f]; mean /= NF;
      for (let f = 0; f < NF; f++) {
        const v = S[t][f]; if (v !== fmax[t][f] || v < mean + 2.0) continue;
        let isMax = true;
        for (let u = Math.max(0, t - RT); u <= Math.min(T - 1, t + RT) && isMax; u++) if (u !== t && fmax[u][f] > v) isMax = false;
        if (isMax) cand.push({ t, f: f + F_LO, v: v - mean });
      }
    }
    const win = Math.round(SR / HOP), out = [];
    for (let w = 0; w < T; w += win) {
      const inWin = cand.filter(p => p.t >= w && p.t < w + win).sort((a, b) => b.v - a.v).slice(0, PEAKS_PER_SEC);
      out.push(...inWin);
    }
    return out.sort((a, b) => a.t - b.t || a.f - b.f);
  }

  // pair each star with the next FANOUT stars in its target zone -> 24-bit hash (f1:9, f2:9, dt:6) + anchor time
  function hashes(P) {
    const H = [];
    for (let i = 0; i < P.length; i++) {
      let n = 0;
      for (let j = i + 1; j < P.length && n < FANOUT; j++) {
        const dt = P[j].t - P[i].t; if (dt < DT_MIN) continue; if (dt > DT_MAX) break;
        if (Math.abs(P[j].f - P[i].f) > DF_MAX) continue;
        H.push([((P[i].f & 511) << 15) | ((P[j].f & 511) << 6) | dt, P[i].t]); n++;
      }
    }
    return H;
  }

  function fingerprint(pcm) { const S = spectrogram(pcm), P = peaks(S); return { S, P, H: hashes(P) }; }

  // library lookup: sorted hash table (Uint32 hashes, Uint16 song, Uint16 time); returns best songs by aligned-offset votes
  function lowerBound(arr, x) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; } return lo; }
  function match(H, db) {
    const votes = new Map();                       // key song*65536 + (offset+32768) -> count
    const hitsBySong = new Map();
    for (const [h, tq] of H) {
      for (let k = lowerBound(db.hash, h); k < db.hash.length && db.hash[k] === h; k++) {
        const s = db.song[k], off = db.time[k] - tq, key = s * 65536 + (off + 32768);
        votes.set(key, (votes.get(key) || 0) + 1);
        if (!hitsBySong.has(s)) hitsBySong.set(s, []); hitsBySong.get(s).push([tq, db.time[k]]);
      }
    }
    const best = new Map();                         // per song: best offset count (+-1 frame tolerance)
    for (const [key, c] of votes) {
      const s = Math.floor(key / 65536), off = (key % 65536) - 32768;
      const tot = c + (votes.get(key - 1) || 0) + (votes.get(key + 1) || 0);
      const b = best.get(s); if (!b || tot > b.score) best.set(s, { song: s, score: tot, offset: off });
    }
    const ranked = [...best.values()].sort((a, b) => b.score - a.score);
    return { ranked, hitsBySong, total: H.length };
  }

  // library file: "SF1" + u32 count, then per entry a varint gap to the previous (sorted) hash and 3 bytes = song (10 bits) | time (14 bits)
  function packDB(rows) {           // rows sorted by hash: [hash, song, time]
    const out = [0x53, 0x46, 0x31, rows.length & 255, (rows.length >> 8) & 255, (rows.length >> 16) & 255, (rows.length >>> 24) & 255]; let prev = 0;
    for (const [h, s, t] of rows) {
      let d = h - prev; prev = h;
      while (d >= 128) { out.push((d & 127) | 128); d >>>= 7; } out.push(d);
      const v = (s << 14) | t; out.push(v & 255, (v >> 8) & 255, (v >> 16) & 255);
    }
    return Uint8Array.from(out);
  }
  function loadDB(buf) {
    const b = new Uint8Array(buf); if (b[0] !== 0x53 || b[1] !== 0x46 || b[2] !== 0x31) throw new Error('not a song library');
    const n = b[3] | (b[4] << 8) | (b[5] << 16) | (b[6] << 24) >>> 0;
    const hash = new Uint32Array(n), song = new Uint16Array(n), time = new Uint16Array(n); let p = 7, prev = 0;
    for (let k = 0; k < n; k++) {
      let d = 0, sh = 0, c; do { c = b[p++]; d += (c & 127) * 2 ** sh; sh += 7; } while (c & 128);
      prev += d; hash[k] = prev; const v = b[p] | (b[p + 1] << 8) | (b[p + 2] << 16); p += 3; song[k] = v >> 14; time[k] = v & 16383;
    }
    return { hash, song, time };
  }

  const API = { SR, N, HOP, F_LO, F_HI, NF, spectrogram, peaks, hashes, fingerprint, match, packDB, loadDB };
  if (typeof module !== 'undefined') module.exports = API; else root.FP = API;
})(typeof window !== 'undefined' ? window : globalThis);
